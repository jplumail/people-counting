#!/bin/bash

export DATA_DIR=/app/data/

tensorboard --logdir=$DATA_DIR/logs --host=0.0.0.0 --load_fast=false &

python train.py &

while true
do
    aws s3 sync /app/data s3://tinyml/data
    sleep 3600
done