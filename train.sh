#!/bin/bash

export S3_REGION=us-east-1

tensorboard --logdir=s3://tinyml/logs --host=0.0.0.0 --load_fast=false &

python train.py