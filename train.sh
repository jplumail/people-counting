#!/bin/bash

export DATA_DIR=/app/data/

tensorboard --logdir=$DATA_DIR/logs --host=0.0.0.0 --load_fast=false &

python train.py &

apt-get install cron

crontab -l > sync
echo "@hourly aws s3 sync /app/data s3://tinyml/data" >> sync
crontab sync
rm sync