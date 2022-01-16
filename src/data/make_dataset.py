# -*- coding: utf-8 -*-
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

parser = argparse.ArgumentParser(description="mnist data constriction arguments")
parser.add_argument("--input_filepath", type=Path, default=os.path.join("data", "raw"))
parser.add_argument(
    "--output_filepath", type=Path, default=os.path.join("data", "processed")
)

args = parser.parse_args()
input_filepath = args.input_filepath
output_filepath = args.output_filepath


def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("reading the training data")
    # initialize list
    images, labels = [], []
    # lopp over training files and append them to lists
    for i in range(5):
        content = np.load(os.path.join(input_filepath, "train_" + str(i) + ".npz"))
        images.append(content["images"])
        labels.append(content["labels"])
    # Transform into tensors
    images = torch.Tensor(np.array(images))
    images = images.flatten(0, 1)
    print(images.shape)
    labels = torch.Tensor(np.array(labels)).type(torch.LongTensor)
    labels = labels.flatten(0, 1)
    # normalize the feature (images)
    images = torch.nn.functional.normalize(images)
    print(images.shape)

    # Save the data
    logger.info("saving the training data")
    torch.save(images, os.path.join(output_filepath, "train_images.pt"))
    torch.save(labels, os.path.join(output_filepath, "train_labels.pt"))

    logger.info("reading the testing data")
    images, labels = [], []

    images = np.load(os.path.join(input_filepath, "test.npz"))["images"]
    labels = np.load(os.path.join(input_filepath, "test.npz"))["labels"]

    images = torch.Tensor(np.array(images))
    labels = torch.Tensor(np.array(labels)).type(torch.LongTensor)

    logger.info("saving the testing data")
    torch.save(images, os.path.join(output_filepath, "test_images.pt"))
    torch.save(labels, os.path.join(output_filepath, "test_labels.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(input_filepath, output_filepath)
