# digit_classification

git clone https://github.com/studious-broccoli/digit_classification.git

# Setup
pip install -e .
pip-compile pyproject.toml --output-file=requirements.txt

# To Test
pytest tests/


# Download data
digit-classification download-data 
--data-dir data/

# Train model
digit-classification train \
--data-dir data/  \
--output-dir checkpoints 
--epochs 20

<figure>
    <img src="images/learning_curve.png" alt="Learning Curve" width="500"/>
    <figcaption>Figure 1: Learning Curve Example.</figcaption>
</figure>

# Evaluate model
digit-classification evaluate \
--data-dir data/ \
--checkpoint-path checkpoints/lightning_logs/version_3/

<figure>
  <img src="images/confusion_matrix_test.png" alt="Confusion Matrix" width="500"/>
  <figcaption>Figure 2: Confusion Matrix Example.</figcaption>
</figure>

# Predict
digit-classification predict \
--checkpoint-path checkpoints/lightning_logs/version_3/ \
--input-path images/test.png

<figure>
    <img src="images/test_prediction.png"  alt="Prediction" width="500"/>
    <figcaption>Figure 3: Prediction Example.</figcaption>
</figure>


