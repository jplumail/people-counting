#!/bin/bash

pip install tensorboard_plugin_profile

export DATA_DIR=/app/data/
export LR=0.0001
export BATCH_SIZE=8

export CHECKPOINT=1632763181
export INITIAL_EPOCH=10001

tensorboard --logdir=$DATA_DIR/ShanghaiTechB/logs --host=0.0.0.0 --load_fast=false &

python train.py &

while true
do
    aws s3 sync /app/data s3://tinyml/data
    sleep 60
done