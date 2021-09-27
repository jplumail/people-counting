#!/bin/bash

pip install tensorboard_plugin_profile

export DATA_DIR=/app/data/
export LR=0.00001

export CHECKPOINT=1632694602
export INITIAL_EPOCH=5143

tensorboard --logdir=$DATA_DIR/ShanghaiTechB/logs --host=0.0.0.0 --load_fast=false &

python train.py &

while true
do
    aws s3 sync /app/data s3://tinyml/data
    sleep 60
done