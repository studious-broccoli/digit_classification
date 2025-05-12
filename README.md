# digit_classification

git clone https://github.com/studious-broccoli/digit_classification.git

# Setup
pip install -e .
pip-compile pyproject.toml --output-file=requirements.txt

# To Test
pytest tests/

# To Run 

# Download data
python src/digit_classification/cli.py download-data 
--data-dir data/

# Train model
python src/digit_classification/cli.py train \
--data-dir data/  \
--output-dir checkpoints 
--epochs 20
![Learning Curve Example](images/learning_curve.png)

# Evaluate model
python src/digit_classification/cli.py evaluate \
--data-dir data/ \
--checkpoint-path checkpoints/lightning_logs/version_3/
![Confusion Matrix Example](images/confusion_matrix_test.png)

# Predict
python src/digit_classification/cli.py predict \
--checkpoint-path checkpoints/lightning_logs/version_3/ \
--input-path images/test.png

![Prediction Example](images/test_prediction.png)


