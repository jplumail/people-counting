#!/bin/bash

# Configure AWS
aws configure

# Log in to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Pull Docker image
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04

# Download data
aws s3 sync s3://tinyml/data/ data

# Start the image
nvidia-docker run -it -p 6006:6006 -v ~/.aws/:/root/.aws/ -v ~/tinyML/:/app/ 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04 bash -c "cd app ; ./train.sh"