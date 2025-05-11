import os
import glob
import pdb

import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


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


def get_latest_ckpt(logs_path="./lightning_logs/version*"):
    files = sorted(glob.glob(f"{logs_path}/**/*.ckpt", recursive=True))
    return files[-1] if files else None


def get_latest_ckpt(logs_path: str = "./lightning_logs/version*"):
    def version_keys(ckpt_dir: str):
        return int(ckpt_dir.split('version_')[-1].split('_')[0].split('/')[0])

    def ckpt_keys(ckpt_fp: str):
        return int(ckpt_fp.split('/')[-1].split('=')[1].split('-')[0])

    latest_ckpts, latest_ckpt_dir = None, None
    log_dirs = sorted(glob.glob(logs_path), key=version_keys, reverse=True)
    for log_dir in log_dirs:
        if os.path.exists(os.path.join(log_dir,"checkpoints")):
            if len(os.listdir(os.path.join(log_dir,"checkpoints"))) > 0:
                checkpoints = sorted(glob.glob(f"{log_dir}/checkpoints/*"), key=ckpt_keys, reverse=True)
                if len(checkpoints) >= 1:
                    latest_ckpt_dir = log_dir
                    latest_ckpts = checkpoints[0]
                    break

    print(f"\n📨 Loading: {latest_ckpt_dir}")
    print(f"📨 Loading: {latest_ckpts}")
    return latest_ckpts


def get_input_dim_num_classes(dataloader):
    for images, labels in dataloader:
        input_shape = images.shape[1:]  # e.g., (1, 28, 28)
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        unique_classes = torch.unique(labels).tolist()
        num_classes = len(set(unique_classes))
        return input_dim, num_classes
