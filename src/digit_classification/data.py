from collections import Counter
from typing import Tuple
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import LabelEncoder
# === Custom Functions ===
from digit_classification.utils.utils import load_config, set_seed
from digit_classification.transforms import data_transform


# Get indices for desired label counts
def get_label_indices(targets, label, count):
    indices = (targets == label).nonzero(as_tuple=True)[0]
    return indices[:count]


# Map select targets to interpretable labels
class MappedSubset(Dataset):
    def __init__(self, subset, label_encoder: LabelEncoder):
        self.subset = subset
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.label_encoder.transform([int(y)])[0]


def get_dataloaders(data_dir: str = "data") -> Tuple[DataLoader, DataLoader, DataLoader]:
    # === Load Config ===
    config = load_config()
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    train_split = config["train_split"]
    target_labels = config["target_labels"]
    seed = config["seed"]

    # === Reproducibility ===
    set_seed(seed)

    # === Load Dataset ===
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=data_transform)
    print("MNIST Dataset Shape =", mnist_dataset.data.shape)
    print(Counter(mnist_dataset.targets.tolist()))

    # === Label Encoder ===
    label_encoder = LabelEncoder()
    label_encoder.fit(target_labels)

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
    mapped_dataset = MappedSubset(subset_dataset, label_encoder)

    # === Split: train / val / test ===
    total_len = len(mapped_dataset)
    train_val_len = int(train_split * total_len)
    test_len = total_len - train_val_len
    train_val_dataset, test_dataset = random_split(mapped_dataset, [train_val_len, test_len])

    train_len = int(train_split * len(train_val_dataset))
    val_len = len(train_val_dataset) - train_len
    train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len])

    print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

    # === DataLoaders ===
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def download_mnist(data_dir: str):
    """
    Downloads the MNIST dataset to the specified directory.
    """
    datasets.MNIST(root=data_dir, train=True, download=True)
    datasets.MNIST(root=data_dir, train=False, download=True)
