import pdb
import os
import glob
import numpy as np
import torch
import smtplib
from typing import Optional
from sklearn.utils.class_weight import compute_class_weight
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ------------------------------
# Extract Version Number
# ------------------------------
def extract_version_number(path: str) -> int:
    """Helper to extract version number from path."""
    try:
        return int(path.split("version_")[-1].split("/")[0])
    except Exception:
        return -1


# ------------------------------
# Extract Epoch Int
# ------------------------------
def extract_epoch_number(ckpt_path: str) -> int:
    try:
        return int(os.path.basename(ckpt_path).split("epoch=")[-1].split("-")[0])
    except Exception:
        return -1


# ------------------------------
# Find Lastest Checkpoint
# ------------------------------
def get_latest_ckpt(logs_root: str = "checkpoints/lightning_logs/version_2") -> Optional[str]:
    """
    Get the latest checkpoint inside a single version directory.
    Expects structure: version_X/checkpoints/*.ckpt
    """
    ckpt_dir = os.path.join(logs_root, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"[WARNING] No 'checkpoints/' directory found in: {logs_root}")
        return None

    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        print(f"[WARNING] No checkpoint files found in: {ckpt_dir}")
        return None

    latest_ckpt = sorted(ckpt_files, key=extract_epoch_number, reverse=True)[0]
    return latest_ckpt


# ------------------------------
# Find Valid Checkpoint from dir or .ckpt
# ------------------------------
def get_valid_checkpoint(checkpoint_path: str) -> str:
    if os.path.isdir(checkpoint_path):
        resume_ckpt = get_latest_ckpt(checkpoint_path)
        if not resume_ckpt:
            raise FileNotFoundError(f"No checkpoint found in: {checkpoint_path}")
    elif checkpoint_path.endswith(".ckpt"):
        resume_ckpt = checkpoint_path
    else:
        raise FileNotFoundError(f"checkpoint_path {checkpoint_path} not in recognizable format.")
    return resume_ckpt


# ------------------------------
# Infer Input Dim and Number of Classes
# ------------------------------
def get_input_dim_num_classes(dataloader):
    for images, labels in dataloader:
        input_shape = images.shape[1:]  # e.g., (1, 28, 28)
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        unique_classes = torch.unique(labels).tolist()
        num_classes = len(set(unique_classes))
        return input_dim, num_classes


# ------------------------------
# Calculate class weights for imbalanced dataset training
# ------------------------------
def calculate_class_weights(train_loader):
    train_labels = []
    for _, y_batch in train_loader:
        train_labels.extend(y_batch.numpy())
    train_labels = np.array(train_labels)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    return class_weights


# ------------------------------
# Send Error Email
# ------------------------------
def send_error_email(subject, body,
                    email_address: str,
                    email_password: str,
                    smtp_server: str,
                    smtp_port: str):
    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = email_address
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.sendmail(email_address, email_address, msg.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")