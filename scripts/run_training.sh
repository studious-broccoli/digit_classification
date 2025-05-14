#!/bin/bash

# Download data
digit-classification  download-data --data-dir data/

# Train model
digit-classification  train --data-dir data/ --output-dir checkpoints --epochs 20