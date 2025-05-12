#!/bin/bash

# Evaluate model
python src/digit_classification/cli.py evaluate \
--data-dir data/MNIST \
--checkpoint-path checkpoints/lightning_logs/version_3/

# Predict
python src/digit_classification/cli.py predict \
--checkpoint-path checkpoints/lightning_logs/version_3/ \
--input-path test.png