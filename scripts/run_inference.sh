#!/bin/bash

# Evaluate model
python src/digit_classification/cli.py evaluate \
--data-dir data/MNIST \
--checkpoint-path checkpoints/lightning_logs/version_5/

## Predict
#python src/digit_classification/cli.py predict \
#--checkpoint-path checkpoints/lightning_logs/version_5/checkpoints/epoch=7-step=504.ckpt \
#--input-path test.png