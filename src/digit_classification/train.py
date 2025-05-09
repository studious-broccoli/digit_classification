from datetime import datetime

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from models.cnn import DigitClassifier
from utils.model_utils import send_error_email, get_latest_ckpt
from utils.utils import load_config

# === Device Setup ===
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# =========================================================
# Constants
# =========================================================
config = load_config()
max_epochs = config["max_epochs"]
load_from_ckpt = config["load_from_ckpt"]
resume_training = ["resume_training"]

EMAIL_ADDRESS = ["EMAIL_ADDRESS"]
EMAIL_PASSWORD = ["EMAIL_PASSWORD"]
SMTP_SERVER = ["SMTP_SERVER"]
SMTP_PORT = ["SMTP_PORT"]


# =========================================================
# Load Data
# =========================================================
input_dim = 0
num_classes = 0

# Define preprocessing (transformations)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Mean & std from MNIST stats
])

# Load training and test datasets
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# =========================================================
# Model
# =========================================================
model = DigitClassifier(input_dim, num_classes)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(every_n_epochs=2, save_top_k=-1)
print("[*] defined monitor and checkpoint callback")

early_stop_callback = EarlyStopping(
    monitor="val_loss",      # or another validation metric
    patience=5,              # number of epochs with no improvement after which training will be stopped
    mode="min",              # "min" for loss, "max" for accuracy, etc.
    verbose=True
)


# =========================================================
# Trainer
# =========================================================
trainer = Trainer(max_epochs=max_epochs,
                  default_root_dir="<path to save logs and checkpoints>",
                  accelerator=device,
                  callbacks=[lr_monitor, checkpoint_callback, early_stop_callback])
print("[*] defined trainer")


# =========================================================
# Training
# =========================================================
if load_from_ckpt:
    latest_ckpt = get_latest_ckpt()
    print(f"[*] found latest ckpt:\n{latest_ckpt}")
else:
    latest_ckpt = None
    print("[*] not loading checkpoint ...")


try:
    if latest_ckpt is not None and resume_training:
        print(f'[INFO] Resuming training from {latest_ckpt}')
        trainer.fit(model, train_loader, ckpt_path=latest_ckpt)
    else:
        print(f'[INFO] Kicking off training from scratch')
        trainer.fit(model, train_loader)
        print("[*] trained model")

except Exception as e:
    # Get the current datetime
    now = datetime.now()
    # Format the datetime string
    datetime_string = now.strftime('%I:%M')
    error_message = f"VICReg Training just DIED at {datetime_string}, \nError: {e}"
    print(error_message)
    send_error_email("VICReg Training just DIED", error_message,
                     EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT)

    log_file_path = f"./error_log_{now}.txt"
    with open(log_file_path, 'a') as file:
        file.write(error_message + '\n')