# -*- coding: utf-8 -*-
import argparse
import logging
import os.path
from pathlib import Path


import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

parser = argparse.ArgumentParser(description="predict arguments")
parser.add_argument(
    "--data_path", type=Path, default=os.path.join("data", "raw", "example_images.npy")
)
parser.add_argument(
    "--model_path", type=Path, default=os.path.join("models", "fcModel.pt")
)


args = parser.parse_args()
data_path = args.data_path
model_path = args.model_path


def main(model_path, data_path):
    """Performs prediction using a model (saved in ./models/model_name.pt)
    using data from data filepath.
    Arguments
            ---------
            model_path: relative path for the model
            data path: relative path for the  data
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading the pretrained model")
    model = torch.load(model_path)
    logger.info("Loading the data")
    images = np.load(data_path)
    images = torch.Tensor(np.array(images))
    images = images.view(images.shape[0], -1)
    logger.info("Predicting the digit")
    logits = model(images)
    ps = torch.exp(logits)
    top_p, top_class = ps.topk(1, dim=1)
    print(top_class)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(model_path, data_path)
