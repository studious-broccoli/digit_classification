import pdb

from digit_classification.data import get_dataloaders, get_testloader
from digit_classification.utils.utils import load_config


def test_get_dataloaders():
    config = load_config()
    data_dir = config["data_dir"]
    train_loader, val_loader = get_dataloaders(data_dir=data_dir)
    test_loader = get_testloader(data_dir=data_dir)

    assert len(train_loader) > 0, "Train DataLoader is empty"
    assert len(val_loader) > 0, "Val DataLoader is empty"
    assert len(test_loader) > 0, "Test DataLoader is empty"

    x, y = next(iter(train_loader))
    x = x.view(x.size(0), -1)  # flatten the input
    assert x.shape[1] == config["input_dim"], "Input data shape mismatch"
    assert len(y) == config["batch_size"], "Batch size mismatch"
