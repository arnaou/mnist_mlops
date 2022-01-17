import os
import pdb
import importlib
import sys

def test_training():
    try:
        from src.models.model import fcModel
    except ImportError or ModuleNotFoundError:
        # The blabla module does not exist, display proper error message and exit
        print('fcModel is not found', file=sys.stderr)
        sys.exit(1)

    if __name__ == "__main__":
        test_training()
