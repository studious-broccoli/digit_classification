import torch
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.utils import load_config


def test_model_forward_pass():
    config = load_config()
    input_dim = config["input_dim"]
    image_dim = config["image_dim"]
    num_channels = config["num_channels"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # === Define Model ===
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)
    # === Create fake input and check output shape ===
    fake_input = torch.randn(batch_size, num_channels, image_dim, image_dim)
    output = model(fake_input)

    assert output.shape == (batch_size, num_classes), (f"Mismatch: Output shape should be [batch_size, num_classes],"
                                                       f" but is {output.shape}")
