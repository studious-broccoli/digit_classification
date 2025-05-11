import os
from pathlib import Path
from digit_classification.train import train_model
from digit_classification.utils.utils import load_config


def test_train_model_runs():
    config = load_config()
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    try:
        train_model(data_dir=data_dir, output_dir=output_dir, epochs=1)
    except Exception as e:
        assert False, f"Training crashed with error: {e}"

    # Confirm checkpoint directory contains some output
    output_dir = Path(config["output_dir"])
    contents = list(output_dir.glob("**/*"))
    assert any("ckpt" in str(p) for p in contents), "No checkpoint file saved"
