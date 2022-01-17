

import torch

from src.models.model import fcModel


def test_model():
    X = torch.rand([1, 28, 28])
    X = X.view(X.shape[0], -1)
    model = fcModel()
    output = model(X)

    assert output.shape == torch.Size([1, 10]), "output does not have the right size"

    if __name__ == "__main__":
        test_model()
