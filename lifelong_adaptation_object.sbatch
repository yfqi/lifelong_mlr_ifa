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
    benchmark_name=LIBERO_OBJECT \
    policy=BriefPolicy \
    lifelong=mlr \
    pretrain=/home/fyu/mlr_ifa/object_multitask_model_ep99.pth \
    +start_task=6 \
    +aug=False \
    +end_task=10 \
    +feature_dir=./features_object \
    +gamma=0.31 \
    +buffer_dir=./buffer_object \
    training=true \
    +past_data_rate=0.5
echo ">>> Training completed ✅"