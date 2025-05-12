from collections import Counter
from typing import Tuple
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data import Dataset
import torch

from digit_classification.utils.utils import load_config, set_seed
from digit_classification.transforms import data_transform


# Get indices for desired label counts
def get_label_indices(targets, label, count):
    indices = (targets == label).nonzero(as_tuple=True)[0]
    return indices[:count]


class MappedSubset(Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.label_map[int(y)]


def get_dataloaders(data_dir: str = "data") -> Tuple[DataLoader, DataLoader, DataLoader]:
    # === Load Config ===
    config = load_config()
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    label_map = config["label_map"]
    seed = config["seed"]

    # === Reproducibility ===
    set_seed(seed)

    # === Load Dataset ===
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=data_transform)
    print("MNIST Dataset Shape =", mnist_dataset.data.shape)
    print(Counter(mnist_dataset.targets.tolist()))

    # === Select target digits ===
    targets = mnist_dataset.targets
    indices = torch.cat([
        get_label_indices(targets, 8, 3500),
        get_label_indices(targets, 0, 1200),
        get_label_indices(targets, 5, 300)
    ])
    indices = indices[torch.randperm(len(indices))]

    # === Subset & remap labels ===
    subset_dataset = Subset(mnist_dataset, indices)
    mapped_dataset = MappedSubset(subset_dataset, label_map)

    # === Split: train / val / test ===
    total_len = len(mapped_dataset)
    train_val_len = int(0.8 * total_len)
    test_len = total_len - train_val_len
    train_val_dataset, test_dataset = random_split(mapped_dataset, [train_val_len, test_len])

    train_len = int(0.8 * len(train_val_dataset))
    val_len = len(train_val_dataset) - train_len
    train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len])

    print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

    # === DataLoaders ===
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_testloader(data_dir="data"):
    config = load_config()
    num_workers = config["num_workers"]
    label_map = config["label_map"]
    batch_size = config["batch_size"]

    # Set the seed at the start
    set_seed(42)

    # Load test dataset
    full_test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=data_transform)
    test_targets = full_test_dataset.targets

    # Filter test indices for labels 0, 5, 8
    test_indices = ((test_targets == 0) | (test_targets == 5) | (test_targets == 8)).nonzero(as_tuple=True)[0]

    # Create subset and DataLoader
    test_dataset = Subset(full_test_dataset, test_indices)
    mapped_dataset = MappedSubset(test_dataset, label_map)
    print("Test Dataset Size =", len(mapped_dataset))

    test_loader = DataLoader(mapped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def download_mnist(data_dir: str):
    """
    Downloads the MNIST dataset to the specified directory.
    """
    datasets.MNIST(root=data_dir, train=True, download=True)
    datasets.MNIST(root=data_dir, train=False, download=True)
