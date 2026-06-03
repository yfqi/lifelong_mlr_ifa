#!/usr/bin/env bash

PYTHON=/home/fyu/miniconda3/envs/libero/bin/python
PORT=$((29500 + RANDOM % 100))

export CUDA_VISIBLE_DEVICES="0"

$PYTHON -m torch.distributed.run \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$PORT \
    libero/lifelong/lifelong_adaptation.py \
    seed=42 \
    benchmark_name=LIBERO_GOAL \
    policy=BriefPolicy \
    lifelong=mlr \
    pretrain=/home/fyu/mlr_ifa/goal_multitask_model_ep96.pth \
    +start_task=6 \
    +aug=False \
    +end_task=10 \
    +feature_dir=./features_goal \
    +gamma=0.72 \
    +buffer_dir=./buffer_goal \
    training=true \
    +past_data_rate=0.5
echo ">>> Training completed ✅"