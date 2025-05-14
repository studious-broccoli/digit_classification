import pdb
from digit_classification.data import get_dataloaders
from digit_classification.utils.utils import load_config


def test_get_dataloaders():
    # === Configuration Settings ===
    config = load_config()
    data_dir = config["data_dir"]
    use_cnn = config["use_cnn"]
    image_dim = config["image_dim"]
    input_dim = config["input_dim"]

    # === Import Dataloaders ===
    train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir)

    assert len(train_loader) > 0, "Train DataLoader is empty"
    assert len(val_loader) > 0, "Val DataLoader is empty"
    assert len(test_loader) > 0, "Test DataLoader is empty"

    x, y = next(iter(train_loader))
    assert len(y) == config["batch_size"], "Batch size mismatch"

    if use_cnn:
        assert x.ndim == 4, f"Expected 4D input [B, C, H, W], got shape {x.shape}"
        assert x.shape[2] == image_dim and x.shape[3] == image_dim, "Image dimensions mismatch"
    else:
        x = x.view(x.size(0), -1)  # flatten the input
        assert x.shape[1] == input_dim, "Input data shape mismatch"