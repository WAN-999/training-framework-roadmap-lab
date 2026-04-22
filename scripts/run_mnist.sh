#!/usr/bin/env bash
set -euo pipefail

python train_mnist.py \
  --epochs 3 \
  --train-batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 2 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --output-dir ./outputs/mnist_exp001