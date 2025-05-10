import os
from digit_classification.data import get_dataloaders


def test_get_dataloaders():
    train_loader, val_loader, test_loader = get_dataloaders(data_dir="data/MNIST")

    assert len(train_loader) > 0, "Train DataLoader is empty"
    assert len(val_loader) > 0, "Val DataLoader is empty"
    assert len(test_loader) > 0, "Test DataLoader is empty"

    sample = next(iter(train_loader))
    assert sample[0].shape[1:] == (1, 28, 28), "Unexpected image shape"
