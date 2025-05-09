# digit_classification

pip install -e .
pip-compile pyproject.toml --output-file=requirements.txt


digit_classification download-data --data-dir ./data
digit_classification train --data-dir ./data --output-dir ./models --model-name resnet18
digit_classification evaluate --checkpoint-path ./models/resnet18.pt --data-dir ./data


rm -rf src/digit_classification.egg-info  # Will regenerate if needed


pip install pip-tools
pip-compile pyproject.toml --output-file=requirements.txt


make install
make train
make test
make requirements
