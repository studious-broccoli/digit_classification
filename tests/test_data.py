from digit_classification.data import get_dataloaders, get_testloader
from digit_classification.utils.utils import load_config

config = load_config()


def test_get_dataloaders(data_dir):
    train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir)
    test_loader = get_testloader(data_dir=data_dir)

    assert len(train_loader) > 0, "Train DataLoader is empty"
    assert len(val_loader) > 0, "Val DataLoader is empty"
    assert len(test_loader) > 0, "Test DataLoader is empty"

    sample = next(iter(train_loader))
    sample_flat = sample.view(sample.size(0), -1)  # flatten the input
    assert sample_flat.shaape[1] == config["INPUT_DIM"], "Input data shape mismatch"
