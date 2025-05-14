#!/bin/bash

# Evaluate model
digit-classification evaluate \
--data-dir data/ \
--checkpoint-path checkpoints/lightning_logs/version_5/

# Create test image
python images/create_test_image.py \
--data-dir data/ \
--image-path images/test.png

# Predict
digit-classification predict \
--checkpoint-path checkpoints/lightning_logs/version_5/ \
--input-path images/test.png