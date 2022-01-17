import os

import pytest
import torch

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    data_type = ["train", "test"]

    for i in data_type:
        images = torch.load(os.path.join(_PATH_DATA, i + "_images.pt"))
        labels = torch.load(os.path.join(_PATH_DATA, i + "_labels.pt"))

        if i == "train":
            n_len = 40000
        elif i == "test":
            n_len = 5000
        assert len(images) == n_len, "not all data are here"
        assert images.shape == torch.Size([n_len, 28, 28]), "input size is wrong"
        assert len(labels) == n_len, "Not all labels are available"

    if __name__ == "__main__":
        test_data()
