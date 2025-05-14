import pdb
from sklearn.preprocessing import LabelEncoder
from digit_classification.data import get_dataloaders
from digit_classification.utils.utils import load_config
from digit_classification.utils.plot_utils import plot_image


def test_get_dataloaders():
    # === Configuration Settings ===
    config = load_config()
    data_dir = config["data_dir"]
    target_labels = config["target_labels"]

    # === Label Encoder ===
    label_encoder = LabelEncoder()
    label_encoder.fit(target_labels)

    train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir)

    assert len(train_loader) > 0, "Train DataLoader is empty"
    assert len(val_loader) > 0, "Val DataLoader is empty"
    assert len(test_loader) > 0, "Test DataLoader is empty"

    x, y = next(iter(train_loader))
    x = x.view(x.size(0), -1)  # flatten the input
    assert x.shape[1] == config["input_dim"], "Input data shape mismatch"
    assert len(y) == config["batch_size"], "Batch size mismatch"

    # Debugging Step (should be updated):
    # Save the first test image once as test.png
    for x_batch, y_batch in test_loader:
        first_image = x_batch[0]  # Tensor of shape [1, 28, 28]
        true_label = label_encoder.inverse_transform([y_batch[0].item()])[0]
        if true_label == 8:
            plot_image(first_image, out_file="images/test.png", title=None)
            print("Saved one test image as images/test.png")
            break  # Only one image for testing right now
