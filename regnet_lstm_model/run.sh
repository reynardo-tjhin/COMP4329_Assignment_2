#!/bin/bash

python3 main.py \
    --model resnet18 \
    --threshold 0.5 \
    --batch_size 16 \
    --learning_rate 0.01 \
    --epochs 20 \
    --log_file log1

# remove any unecessary temporary files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|instance)" | xargs rm -rf