import os
import glob
import pdb

import torch
import smtplib
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ------------------------------
# Extract Version Number
# ------------------------------
def extract_version_number(path: str) -> int:
    try:
        return int(path.split("version_")[-1].split('/')[0])
    except (IndexError, ValueError):
        return -1  # fallback for malformed paths


# ------------------------------
# Extract Epoch Int
# ------------------------------
def extract_epoch_number(ckpt: str) -> int:
    try:
        return int(ckpt.split("epoch=")[-1].split('-')[0])
    except (IndexError, ValueError):
        return -1


# ------------------------------
# Find Lastest Checkpoint
# ------------------------------
def get_latest_ckpt(logs_root: str = "./lightning_logs") -> Optional[str]:
    log_dirs = sorted(
        glob.glob(os.path.join(logs_root, "version*")),
        key=extract_version_number,
        reverse=True,
    )

    for log_dir in log_dirs:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            if ckpt_files:
                return sorted(ckpt_files, key=extract_epoch_number, reverse=True)[0]
    return None


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