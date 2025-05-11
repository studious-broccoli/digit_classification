#!/bin/bash

## Evaluate model
#python src/digit_classification/cli.py evaluate \
#--data-dir data/MNIST \
#--checkpoint-path checkpoints/lightning_logs/version_3/

# Predict
python src/digit_classification/cli.py predict \
--checkpoint-path checkpoints/lightning_logs/version_3/checkpoints/epoch=9-step=630.ckpt \
--input-path test.png