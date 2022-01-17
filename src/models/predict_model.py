# -*- coding: utf-8 -*-
import argparse
import logging
import os.path
from pathlib import Path


import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
import hydra
import logging
from omegaconf import OmegaConf
import pdb

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def prediction(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams_pred = config.experiment_predict
    #pdb.set_trace()

    log.info("Loading the pretrained model")
    model = torch.load(hparams_pred['model_path'])

    log.info("Loading the data")
    images = np.load(hparams_pred['data_path'])
    images = torch.Tensor(np.array(images))

    images = images.view(images.shape[0], -1)
    log.info("Predicting the digit")
    logits = model(images)
    ps = torch.exp(logits)
    top_p, top_class = ps.topk(1, dim=1)
    print(top_class)


if __name__ == "__main__":
    prediction()
# def main(model_path, data_path):
#     """Performs prediction using a model (saved in ./models/model_name.pt)
#     using data from data filepath.
#     Arguments
#             ---------
#             model_path: relative path for the model
#             data path: relative path for the  data
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Loading the pretrained model")
#     model = torch.load(model_path)
#     logger.info("Loading the data")
#     images = np.load(data_path)
#     images = torch.Tensor(np.array(images))
#     images = images.view(images.shape[0], -1)
#     logger.info("Predicting the digit")
#     logits = model(images)
#     ps = torch.exp(logits)
#     top_p, top_class = ps.topk(1, dim=1)
#     print(top_class)

#
# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)
#
#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]
#
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())
#
#     main(model_path, data_path)
