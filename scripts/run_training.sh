# Download data
python src/cli.py download-data --data-dir ./data

# Train model
python src/cli.py train --data-dir ./data --output-dir ./models --model-name resnet18 --epochs 10

# Evaluate model
python src/cli.py evaluate --checkpoint-path ./models/resnet18.pt --data-dir ./data --model-name resnet18

# Predict
python src/cli.py predict --checkpoint-path ./models/resnet18.pt --input-path ./test_digit.png --model-name resnet18
