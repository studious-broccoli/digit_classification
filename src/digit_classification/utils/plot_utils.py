import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import torch


# ------------------------------
# Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name="CNN", dataset="Validation", filename="cm.png", class_names=None):
    # Auto-detect class names if not provided
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))

    num_classes = len(class_names)
    cf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)

    # Accuracy & F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    title = f"{model_name}\n{dataset} Accuracy: {acc*100:.2f}%, F1-Score: {f1*100:.2f}%\n(N = {len(y_true)})"

    # Specificity
    total_preds = np.sum(cf_matrix)
    tn = total_preds - cf_matrix.sum(axis=1)
    tp = np.diag(cf_matrix)
    fp = cf_matrix.sum(axis=0) - tp
    specificity = tn / (tn + fp)

    # Recall
    denom = cf_matrix.sum(axis=1)[:, None]
    recall = np.divide(cf_matrix, denom, out=np.zeros_like(cf_matrix, dtype=float), where=denom!=0)

    # Format annotations
    diag_indx = [i * (num_classes + 1) for i in range(num_classes)]
    group_counts = [f"{val:0.0f}" for val in cf_matrix.flatten()]
    group_recall = ["" if i not in diag_indx or np.isnan(v) else f"Se: {v:.1%}" for i, v in enumerate(recall.flatten())]
    group_specificity = ["" if i not in diag_indx else f"Sp: {specificity[i % num_classes]:.1%}" for i in range(num_classes ** 2)]
    labels = [f"{c}\n{r}\n{s}" for c, r, s in zip(group_counts, group_recall, group_specificity)]
    labels = np.array(labels).reshape(num_classes, num_classes)

    # Plot
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_xlabel('\nPredicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names, rotation=45)
    plt.title(title)

    # Save and close
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return title


# ------------------------------
# Format Uniform CM Title
# ------------------------------
def format_cm_title(y_true, y_pred, model_name="CNN", dataset="Validation"):
    acc = accuracy_score(y_true, y_pred),
    f1 = f1_score(y_true, y_pred, average="weighted")

    cm_title = (
        f"{model_name}\n{dataset} Accuracy: {acc * 100:.2f}%, "
        f"F1-Score: {f1 * 100:.2f}%\n(N = {len(y_true)})"
    )
    return cm_title


# ------------------------------
# Classification Report
# ------------------------------
def print_classification_report(y_true, y_pred, message=None):
    if message:
        print("\n", message)
    print(classification_report(y_true, y_pred))


# ------------------------------
# Plot Test Instance
# ------------------------------
def plot_image(image, out_file, title=""):
    plt.figure()

    # Handle grayscale tensors or arrays
    if isinstance(image, torch.Tensor):
        image = image.squeeze().detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.squeeze()

    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


# ------------------------------
# Plot Learning Curve for Training
# ------------------------------
def plot_learning_curves(log_dir: str) -> None:
    """Plot training and validation loss curves from CSV logs."""
    csv_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print("[WARNING] No metrics.csv found — skipping plot.")
        return

    df = pd.read_csv(csv_path)
    if 'epoch' not in df.columns:
        print("[WARNING] Invalid metrics file format.")
        return

    train_loss = df[df["split"] == "train"]
    val_loss = df[df["split"] == "val"]

    plt.figure()
    if 'loss' in train_loss.columns:
        plt.plot(train_loss["epoch"], train_loss["loss"], label="Train Loss")
    if 'loss' in val_loss.columns:
        plt.plot(val_loss["epoch"], val_loss["loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(log_dir, "learning_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved learning curve to {out_path}")