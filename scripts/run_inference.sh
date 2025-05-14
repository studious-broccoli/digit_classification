#!/bin/bash

# Evaluate model
digit-classification evaluate \
--data-dir data/ \
--checkpoint-path checkpoints/lightning_logs/version_5/

# Predict
digit-classification predict \
--checkpoint-path checkpoints/lightning_logs/version_5/ \
--input-path images/test.png