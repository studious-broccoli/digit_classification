import yaml
import random
import torch
import numpy as np


def load_config(file_path="./configs/config.yaml"):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # Load YAML file safely
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For determinism in convolutional algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

