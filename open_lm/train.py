import itertools
import logging
import math
import time
from contextlib import nullcontext
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


try:
    from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss
    from megablocks.layers.arguments import Arguments as MoEArgs
except ImportError:
    logging.warning(f"Megablocks not installed. To train MoE, install with pip install megablocks.")

try:
    import wandb
except ImportError:
    wandb = None

from open_lm.distributed import is_master
from open_lm.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfidenceIntervalMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.points = []
        self.points_tensor = None

    def update(self, val):
        self.points.append(val)

    def compute_bootstrap_ci(self, num_samples=10_000, interval=95):
        lower = None
        upper = None

        self.points_tensor = torch.cat(self.points)
        num_points = self.points_tensor.shape[0]

        estimates = []
        for _ in range(num_samples):
            i = np.random.choice(num_points, size=num_points)
            estimate = torch.sum(self.points_tensor[i]) / num_points
            estimates.append(estimate.item())

        half = (100 - interval) / 2

        lower = np.percentile(estimates, half).item()
        upper = np.percentile(estimates, 100 - half).item()

        return lower, upper


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def replace_before_tok(tensor, tok, replaced, excusive=False):
    # NOTE: this implementation supports 0 or 1 instance of tok in a sequence.
    #       if more than one instance appears, the last instace of tok is used.
    #       if exclusive=True every instance of tok will be present in the output

    tok_positions = tensor == tok

    # construct cumulative mask for positions before (last) tok (if it appears)
    cumsum_mask = tok_positions.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    # create mask for positions before (last) tok in each row (batch)
    tok_mask = cumsum_mask > 0

    if excusive:
        # retain tok in the output
        tok_mask &= ~tok_positions

    out = torch.clone(tensor)
    out[tok_mask] = replaced

    return out


def replace_tok(tensor, tok, replaced):
    out = torch.clone(tensor)
    out[out == tok] = replaced

    return out


def sample_chunk(chunk, args):
    if chunk.shape[1] == args.seq_len + 1:
        start_idx = 0
    elif chunk.shape[1] > args.seq_len + 1:
        start_idx = torch.randint(0, chunk.shape[1] - args.seq_len, (1,)).item()
    else:
        raise Exception(f"Invalid sequence length: Sequence length {args.seq_len} > {chunk.shape[1]} Chunk size")

    inputs = chunk[:, start_idx : start_idx + args.seq_len]
    targets = chunk[:, start_idx + 1 : start_idx + args.seq_len + 1]

    # replace elements to be masked with with -100 (pytorch default xent ignore value)
    if args.target_mask_left is not None:
        targets = replace_before_tok(targets, args.target_mask_left, -100)
    if args.target_mask_individual is not None:
        targets = replace_tok(targets, args.target_mask_individual, -100)

    return inputs, targets


def train_one_epoch(model, data, loss, epoch, step, optimizer, scaler, scheduler, total_steps, args, tb_writer=None):
    """Trains model for one epoch on the provided data.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the epoch. Note that "epoch" actually is not one full pass through the
            data, but rather the number of tokens specified by `--train-num-samples`, rounded based on shard size.
            As such, the number of steps in an "epoch" can vary, and we have to keep track of steps separately.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    data["train"].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    load_balancing_losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()

    end = time.time()

    data_iterator = iter(dataloader)

    if args.moe_freq > 0:
        # these MoEArgs are necessary for logging load balancing.
        moe_args = MoEArgs(
            hidden_size=model.dim,
            ffn_hidden_size=model.dim * 4,
            moe_num_experts=args.moe_num_experts,
            num_layers=model.n_layers // 2,
            moe_expert_model_parallelism=True,
            moe_top_k=args.moe_top_k,
            device=torch.cuda.current_device(),
            moe_capacity_factor=args.moe_capacity_factor,
            moe_loss_weight=args.moe_loss_weight,
            fp16=False,
            bf16=False,
        )

    for i in itertools.count():
        if not args.skip_scheduler:
            scheduler(step)

        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        if args.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        if has_data < args.world_size:
            break

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                inputs, targets = sample_chunk(texts, args)
                out, _, _ = model(inputs)

                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())

                total_lm_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
                total_loss = total_lm_loss
                if args.moe_freq > 0:
                    total_load_balancing_loss = batched_load_balancing_loss(moe_args)
                    clear_load_balancing_loss()
                    total_loss += total_load_balancing_loss

            backward(total_loss, scaler)
        else:
            # split up batch into accum_freq chunks -- if you have --batch-size 8 and --accum-freq 4
            # then you only process 2 items at a time. batch-size must be divisible by accume-freq.
            assert args.per_gpu_batch_size % args.accum_freq == 0, "Per-GPU batch size must be divisible by accum_freq"
            per_batch = args.per_gpu_batch_size // args.accum_freq

            inputs, targets = sample_chunk(texts, args)

            for ii in range(args.accum_freq):
                maybe_no_sync = nullcontext
                # Don't sync gradients until the final batch for FSDP.
                if isinstance(model, FSDP) and ii != args.accum_freq - 1:
                    maybe_no_sync = model.no_sync
                with maybe_no_sync():
                    with autocast():
                        inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                        if inputs_ii.shape[0] == 0:
                            break
                        targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]
                        out, _, _ = model(inputs_ii)

                        if args.log_logit_mean:
                            logit_m.update(torch.mean(out).item())

                        local_lm_loss = (
                            loss(out.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                            * inputs_ii.shape[0]
                            / inputs.shape[0]
                        )
                    local_loss = local_lm_loss
                    if args.moe_freq > 0:
                        local_load_balancing_loss = batched_load_balancing_loss(moe_args)
                        clear_load_balancing_loss()
                        local_loss += local_load_balancing_loss

                    backward(local_loss, scaler)
                if ii == 0:
                    total_lm_loss = local_lm_loss
                    if args.moe_freq > 0:
                        total_load_balancing_loss = local_load_balancing_loss
                else:
                    total_lm_loss += local_lm_loss
                    if args.moe_freq > 0:
                        total_load_balancing_loss += local_load_balancing_loss

            total_loss = total_lm_loss
            if args.moe_freq > 0:
                total_loss += total_load_balancing_loss

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()

        global_loss_tensor = total_loss.detach().clone()
        if args.world_size > 1:
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)

        batch_count = i + 1
        step += 1

        if is_master(args) and (
            i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch or step == total_steps - 1
        ):
            batch_size = len(inputs)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # gathered_loss = [torch.zeros_like(total_loss) for _ in range(args.world_size)]
            # torch.distributed.all_gather(gathered_loss, total_loss)

            # losses_m.update(sum(gathered_loss).item() / args.world_size, batch_size * args.world_size)
            if args.moe_freq > 0:
                losses_m.update(global_loss_tensor.item() - total_load_balancing_loss.item(), batch_size)
                load_balancing_losses_m.update(total_load_balancing_loss.item(), batch_size)
            else:
                losses_m.update(global_loss_tensor.item(), batch_size)
            samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
            loss_str = f"Loss: {losses_m.avg:.3f}"
            loss_str += f" LB-Loss: {load_balancing_losses_m.avg:.3f}" if args.moe_freq > 0 else ""
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"{loss_str} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": losses_m.val,
                "load_balancing_loss": load_balancing_losses_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens": (step + 1) * args.global_batch_size * args.seq_len,
            }

            if args.log_logit_mean:
                log_data["logit_mean"] = logit_m.val

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": step, "tokens": log_data["tokens"]})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

            if math.isnan(losses_m.val):
                # case where loss goes to nan, we see this sometimes with bad nodes.
                # in this case we would like to free resources and prevent other issues
                # e.g., saving checkpoints and optmization states that may lead to skipped
                # training on restarts.
                return False, step

    # end for
    return True, step


@torch.inference_mode()
def evaluate(model, data, start_epoch, args, writer):
    """
    evaluates perplexity on validation data
    """
    if is_master(args):
        print("=> begin evaluation")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    data["val"].set_epoch(start_epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["val"].dataloader

    # NOTE: max_num_batches = 0 corresponds to exhausting iterator
    max_num_batches = dataloader.num_batches

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    sps_m = AverageMeter()
    spspg_m = AverageMeter()
    losses_seq_ci_m = ConfidenceIntervalMeter()
    losses_tok_ci_m = ConfidenceIntervalMeter()

    end = time.time()
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    for i, batch in enumerate(dataloader):
        if i == max_num_batches and max_num_batches != 0:
            break

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)

        data_time_m.update(time.time() - end)

        with autocast():
            inputs, targets = sample_chunk(texts, args)

            out, _, _ = model(inputs)  # [bs, seq_len, vocab_size]

            bs, seq_len = targets.shape

            targets = targets.reshape(-1)
            total_loss = loss(out.reshape(-1, args.vocab_size), targets)  # [bs * seq_len]

            # cross entropy ignores -100 values in loss computation
            mask = targets != -100

            # reshape and average for sequence losses
            sum_loss_per_seq = torch.sum(total_loss.reshape(bs, seq_len), -1)
            num_toks_per_seq = torch.sum(mask.reshape(bs, seq_len), -1).float()
            losses_seq_ci_m.update(sum_loss_per_seq / num_toks_per_seq)

            # individual token losses
            losses_tok_ci_m.update(total_loss[mask])

            # compute average loss for the mini-batch
            total_loss = total_loss[mask].mean()
            losses_m.update(total_loss.item(), n=inputs.shape[0])

        batch_time_m.update(time.time() - end)
        sps_m.update(inputs.numel() * args.world_size / batch_time_m.val)
        spspg_m.update(inputs.numel() / batch_time_m.val)

    lower_seq, upper_seq = losses_seq_ci_m.compute_bootstrap_ci()
    lower_tok, upper_tok = losses_tok_ci_m.compute_bootstrap_ci()
    num_seqs = losses_seq_ci_m.points_tensor.shape[0]
    num_toks = losses_tok_ci_m.points_tensor.shape[0]

    # Save eval loss / etc.
    log_data = {
        "loss": losses_m.avg,
        "data_time": data_time_m.avg,
        "batch_time": batch_time_m.avg,
        "samples_per_second": sps_m.avg,
        "samples_per_second_per_gpu": spspg_m.avg,
        "loss_sequences_lower_95": lower_seq,
        "loss_sequences_upper_95": upper_seq,
        "loss_tokens_lower_95": lower_tok,
        "loss_tokens_upper_95": upper_tok,
        "sequences": num_seqs,
        "tokens": num_toks,
    }
    if args.train_num_samples is not None:
        log_data["train_tokens"] = start_epoch * args.train_num_samples * args.seq_len

    for name, val in log_data.items():
        name = "valid/" + name
        if writer is not None:
            writer.add_scalar(name, val, start_epoch)
        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            wandb.log({name: val, "epoch": start_epoch, "tokens": log_data["tokens"]})
    if is_master(args):
        print(f"evaluation on: {args.val_data}")
        print(f"evaluation loss: {losses_m.avg}")
        print(f"evaluation perplexity: {math.exp(losses_m.avg)}")
        print(f"num seqs: {num_seqs}")
        print(f"num tokens: {num_toks}")

    log_data["checkpoint_path"] = args.resume
    log_data["val_data"] = args.val_data
    log_data["model"] = args.hf_model if args.hf_model else args.model

    return log_data


def evaluate_loop(model, data_list, start_epoch, args, writer):
    log_data_list = []
    for i, data in enumerate(data_list):
        args_copy = copy.deepcopy(args)
        args_copy.val_data = [args.val_data[i]]
        args_copy.val_data_key = args.val_data_key[i]

        log_data_list.append(evaluate(model, data, start_epoch, args_copy, writer))

    return log_data_list
