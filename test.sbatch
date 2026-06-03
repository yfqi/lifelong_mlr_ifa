#!/usr/bin/env bash

PYTHON=/home/fyu/miniconda3/envs/libero/bin/python

export CUDA_VISIBLE_DEVICES="0"

export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH=/home/fyu/mlr_ifa:$PYTHONPATH

$PYTHON libero/lifelong/test.py \
    seed=42 \
    benchmark_name=LIBERO_OBJECT \
    policy=complete_policy \
    lifelong=basic \
    pretrain=/home/fyu/mlr_ifa/object_multitask_model_ep99.pth  \
    +start_task=6 \
    +end_task=10 \
    +aug=True \
    training=False \
    +epoch_dir=./epoch \
    +test_log_file=./log.txt

echo ">>> Training completed ✅"