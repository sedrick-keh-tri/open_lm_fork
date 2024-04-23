# TORCH_DISTRIBUTED_DEBUG="DETAIL" torchrun --nproc-per-node 3 -m open_lm.main \
CUDA_VISIBLE_DEVICE=1 CUDA_LAUNCH_BLOCKING=1 python open_lm/main.py \
 --model linear_tiny \
 --dataset-manifest s3://tri-ml-datasets/openlm/dcnlp/datasets/rpj_lmdata_do_sample_weight_fix_try2/manifest.jsonl \
 --train-num-samples 1_000_000 \
 --precision "amp_bfloat16" \
 --fsdp-amp \
 --fsdp-pure-bf16 \
 --workers 1 \
 --global-batch-size 9 \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key json.gz \
 --lr 3e-4 \
 --accum-freq 1 \
 --warmup 10 \
 --wd 0.1 \
 --beta2 0.98 \
 --epochs 10 \
 --report-to wandb \
 --wandb-project-name open_lm \
 --name open_lm_ex_$RANDOM \
 --resume latest \
 --logs logs \
 --z-loss-coefficient 1e-4 \
 --load-not-strict \

 

