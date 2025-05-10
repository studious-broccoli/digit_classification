import torch
from digit_classification.models.cnn import DigitClassifier


def test_model_forward_pass():
    model = DigitClassifier(input_dim=28 * 28, num_classes=10)
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)

    assert output.shape == (8, 10), "Output shape should be [batch_size, num_classes]"
