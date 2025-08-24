#!/bin/bash
#SBATCH --job-name=fine_tuning
#SBATCH --output=logs/logs_%j.out
#SBATCH --partition=frida
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:B200:4
#SBATCH --time=72:00:00
#SBATCH --mem=128G

CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

srun \
    --cpu-bind=verbose \
    --container-image $CONTAINER \
    --container-mounts $HOME/translate_env:/mnt/env,/shared/workspace/povejmo:/workspace \
    --container-workdir /workspace/translation_optimization/other/miniGPT/miniVLM \
        bash -lc "torchrun --nproc_per_node=1 --master_port=29500 ddp_pretrain.py --output=test_run --should_checkpoint --checkpoint_every 10000"
