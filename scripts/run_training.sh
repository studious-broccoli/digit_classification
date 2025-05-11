#!/bin/bash

# Download data
python src/digit_classification/cli.py download-data --data-dir data/MNIST

# Train model
python src/digit_classification/cli.py train --data-dir data/MNIST --output-dir checkpoints --epochs 20