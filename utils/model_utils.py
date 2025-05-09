import glob
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