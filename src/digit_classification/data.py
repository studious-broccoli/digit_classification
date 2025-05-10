# python src/digit_classification/cli.py download-data --data-dir data/MNIST
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch

import os
NUM_WORKERS = 0 #min(2, os.cpu_count() or 1)  # Safe for macOS

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Get indices for desired label counts
def get_label_indices(targets, label, count):
    indices = (targets == label).nonzero(as_tuple=True)[0]
    return indices[:count]


def get_dataloaders(data_dir="data"):
    # Load full MNIST dataset
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    print("MNIST Dataset Shape =", mnist_dataset.data.shape)
    print(Counter(mnist_dataset.targets.tolist()))

    # Access all targets
    targets = mnist_dataset.targets

    label_8_indices = get_label_indices(targets, 8, 3500)
    label_0_indices = get_label_indices(targets, 0, 1200)
    label_5_indices = get_label_indices(targets, 5, 300)

    # Combine and shuffle indices
    selected_indices = torch.cat([label_8_indices, label_0_indices, label_5_indices])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]

    # Create subset dataset
    subset_dataset = Subset(mnist_dataset, selected_indices)

    # 80/20 split
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    # Load test dataset
    full_test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    test_targets = full_test_dataset.targets

    # Filter test indices for labels 0, 5, 8
    test_indices = ((test_targets == 0) | (test_targets == 5) | (test_targets == 8)).nonzero(as_tuple=True)[0]

    # Create subset and DataLoader
    test_dataset = Subset(full_test_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader


def download_mnist(data_dir: str):
    """
    Downloads the MNIST dataset to the specified directory.
    """
    datasets.MNIST(root=data_dir, train=True, download=True)
    datasets.MNIST(root=data_dir, train=False, download=True)
