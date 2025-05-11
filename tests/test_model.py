import os
import shutil
from digit_classification.train import train_model


def test_train_model_runs(tmp_path):
    # Use a small, temporary checkpoint dir for test
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir()

    try:
        train_model(data_dir="data/MNIST", output_dir=str(output_dir), epochs=1)
    except Exception as e:
        assert False, f"Training crashed with error: {e}"

    # Confirm checkpoint directory contains some output
    contents = list(output_dir.glob("**/*"))
    assert any("ckpt" in str(p) for p in contents), "No checkpoint file saved"
