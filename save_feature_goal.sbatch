#!/usr/bin/env bash
set -e
export HYDRA_FULL_ERROR=1

PYTHON=/home/fyu/miniconda3/envs/libero/bin/python
PORT=$((29500 + RANDOM % 100))

$PYTHON -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:$PORT \
  libero/lifelong/save_feature.py \
  seed=42 \
  benchmark_name=LIBERO_GOAL \
  policy=complete_policy \
  lifelong=basic \
  train.batch_size=1 \
  pretrain=./multitask_model_ep96.pth \
  +start_task=6 \
  +end_task=10 \
  +aug=True \
  +feature_dir=./features_goal \
  training=False

echo ">>> Save feature completed ✅"