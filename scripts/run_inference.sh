# Evaluate model
python src/digit_classification/cli.py evaluate \
--data-dir data/MNIST \
--checkpoint-path checkpoints/lightning_logs/version_0/checkpoints/epoch=9-step=630.ckpt

# Predict
python src/digit_classification/cli.py predict \
--checkpoint-path checkpoints/lightning_logs/version_0/checkpoints/epoch=9-step=630.ckpt \
--input-path <path to input image>

#!/bin/bash