import sys

import torch


def test_training():
    try:
        from src.models.model import fcModel
        model = fcModel()
        X = torch.rand([1, 28, 28])
        X = X.view(X.shape[0], -1)
        model(X)
    except ImportError or ModuleNotFoundError:
        # The blabla module does not exist, display proper error message and exit
        print('fcModel is not found', file=sys.stderr)
        sys.exit(1)

    if __name__ == "__main__":
        test_training()
