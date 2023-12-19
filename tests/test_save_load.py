import pytest
import argparse
import torch
import os

import torch.multiprocessing as mp

from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.model import create_params
from open_lm.train import train_one_epoch
from open_lm.main import save_checkpoint, load_model
from open_lm.losses import CrossEntropyLossWithZLoss
from open_lm.distributed import is_using_distributed

from tests.shared import MockTrainArgs, create_train_fixtures
from tests.utils import download_dl_test_data


@pytest.fixture(scope="module")
def tiny_args():
    args = MockTrainArgs(
        model="open_lm_test_tiny",
        **{
            "vocab_size": 16,
            "sequence_length": 16,
            "train_num_samples": 64,
            "batch_size": 4,
            # Model params that might not be in config:
            "model_norm": "default_layer_norm",
            "qk_norm": False,
            "positional_embedding_type": "rotary",
            "ffn_type": "swiglu",
        }
    )
    return args


def test_tiny_save_load(tiny_args, fsdp=False):
    """
    This test checks that the model can be saved and loaded without changing the parameters.
    """
    scaler = None
    epoch = 0
    evaluation_metrics = None
    global_step = 0
    done_training = False
    download_dl_test_data()
    override_params = dict(
        seq_len=tiny_args.sequence_length,
        vocab_size=tiny_args.vocab_size,
        train_num_samples=tiny_args.train_num_samples,
        global_batch_size=tiny_args.global_batch_size,
        checkpoint_path="./tests/assets/checkpoints/tiny_model/",
        device="cpu" if not torch.cuda.is_available() else "cuda",
        dataset_type="synthetic",
        save_logs=True,
    )

    args, model, data, optimizer, scheduler, loss = create_train_fixtures(
        tiny_args.model, fsdp=False, **override_params
    )
    model = model.to(args.device)
    args2, model2, data2, optimizer2, scheduler2, loss2 = create_train_fixtures(
        tiny_args.model, fsdp=False, **override_params
    )
    model2 = model2.to(args2.device)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # print("Training tiny model")
    train_one_epoch(
        model,
        data,
        CrossEntropyLossWithZLoss(),
        epoch=epoch,
        step=global_step,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        total_steps=args.train_num_samples // args.global_batch_size,
        args=args,
    )
    epoch += 1
    threshold = 1e-6
    # Checking that tiny models diverged after traning.
    allclose = True
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        allclose = allclose and torch.allclose(p1, p2, atol=threshold)
    assert not allclose

    args.distributed = is_using_distributed()
    # print("Saving tiny model")
    # Saving checkpoints.
    save_checkpoint(
        args,
        model,
        optimizer,
        scaler,
        epoch,
        evaluation_metrics,
        step=global_step,
        is_final_checkpoint=done_training,
        next_shard_per_source=None,
        samples_seen=None,
    )

    # Loading saved tiny model
    args.resume = "./tests/assets/checkpoints/tiny_model/epoch_1.pt"
    load_model(args, model2)

    # Checking that loaded tiny model is the same as the original tiny model
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=threshold)


def _save_load_helper_fsdp(rank, world_size, tiny_args):
    # Initialize distributed training
    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29501",
        rank=rank,
        world_size=world_size,
    )
    test_tiny_save_load(tiny_args, fsdp=True)
    torch.distributed.destroy_process_group()


def test_tiny_save_load_fsdp(tiny_args):
    world_size = 1
    mp.spawn(_save_load_helper_fsdp, args=(world_size, tiny_args), nprocs=world_size, join=True)


if __name__ == "__main__":
    pytest.main([__file__])
