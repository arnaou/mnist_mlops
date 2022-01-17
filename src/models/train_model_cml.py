# -*- coding: utf-8 -*-

import logging
import os.path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from model import fcModel
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    wandb.init(config=config, project='mnist_arnaou', entity='arnaou')
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams_mod = config.experiment_model
    hparams_tr = config.experiment_train

    torch.manual_seed(hparams_tr["seed"])
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    log.info("Initializing the Model")
    model = fcModel(hparams_mod["input_size"], hparams_mod["output_size"], hparams_mod["hidden_layers"],
                    hparams_mod['dropout_rate'])
    model = model.to(DEVICE)
    wandb.watch(model, log_freq=100)

    log.info("Loading processed training data")
    images = torch.load(os.path.join(hparams_tr["data_path"], "train_images.pt"))
    labels = torch.load(os.path.join(hparams_tr["data_path"], "train_labels.pt"))
    train_set = DataLoader(TensorDataset(images, labels), batch_size=hparams_tr["Batch_size"], shuffle=True)

    log.info("Training the model")
    # define criterion
    criterion = nn.NLLLoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams_tr["lr"])
    # Bookkeeping for loss and steps
    steps = 0
    loss_list = []
    targets. preds = [], []
    # loop over epochs
    for e in range(hparams_tr["n_epochs"]):
        # loop over batches
        running_loss = 0
        for images, labels in train_set:
            steps += 1
            # flatten images
            images = images.view(images.shape[0], -1)
            # set optimizer gradient to 0
            optimizer.zero_grad()
            # Evaluate model: forward pass
            output = model(images)
            # calculate loss
            loss = criterion(output, labels)
            # Calculate gradients
            loss.backward()
            # update the weights
            optimizer.step()
            # calculate loss
            running_loss += loss.item()
            # make list of predictions and targets
            preds.append(output.argmax(dim=-1))
            targets.append(output.detach())

        loss_list.append(running_loss)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)


        wandb.log({"loss": running_loss})



    # saving the model
    log.info("Saving the model")

    torch.save(model, hparams_tr["model_path"])
    # plotting the learning curve
    plt.plot(np.arange(hparams_tr["n_epochs"]), np.array(loss_list))
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    log.info("Saving the plot")
    plt.savefig(hparams_tr["report_path"])
    wandb.log({"chart": wandb.Image(plt)})


if __name__ == "__main__":
    train()
#
#
# parser = argparse.ArgumentParser(description="training arguments")
# parser.add_argument("--data_path", type=Path, default=os.path.join("data", "processed"))
# parser.add_argument("--input_size", type=int, default=784)
# parser.add_argument("--output_size", type=int, default=10)
# parser.add_argument("--hidden_layers", nargs="+", type=int, default=[512, 256, 128])
# parser.add_argument("--dropout_rate", type=float, default=0.2)
#
# args = parser.parse_args()
# data_filepath = args.data_path
# in_size = args.input_size
# out_size = args.output_size
# hidd_layers = args.hidden_layers
# dropout_rate = args.dropout_rate
#
#
# def main(data_filepath, in_size, out_size, hidd_layers, dropout_rate):
#     """Trains a model using data from (../processed) and saves a model into
#     (saved in ./models/model_name.pt).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Loading Model")
#     model = fcModel(in_size, out_size, hidd_layers, dropout_rate)
#     logger.info("Loading processed training data")
#     images = torch.load(os.path.join(data_filepath, "train_images.pt"))
#     labels = torch.load(os.path.join(data_filepath, "train_labels.pt"))
#     train_set = DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True)
#     logger.info("Training the model")
#
#     # define criterion
#     criterion = nn.NLLLoss()
#     # define optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
#     # Bookkeeping for loss and steps
#     steps = 0
#     loss_list = []
#     # define nr of epochs
#     epochs = 5
#     # loop over epochs
#     for e in range(epochs):
#         # loop over batches
#         running_loss = 0
#         for images, labels in train_set:
#             steps += 1
#             # flatten images
#             images = images.view(images.shape[0], -1)
#             # set optimizer gradient to 0
#             optimizer.zero_grad()
#             # Evaluate model: forward pass
#             output = model(images)
#             # calculate loss
#             loss = criterion(output, labels)
#             # Calculate gradients
#             loss.backward()
#             # update the weights
#             optimizer.step()
#             # calculate loss
#             running_loss += loss.item()
#
#         loss_list.append(running_loss)
#     # saving the model
#     logger.info("Saving the model")
#     torch.save(model, os.path.join("models", "fcModel.pt"))
#     # plotting the learning curve
#     plt.plot(np.arange(epochs), np.array(loss_list))
#     plt.title("Learning Curves")
#     plt.xlabel("Epochs")
#     plt.ylabel("Training loss")
#     logger.info("Saving the plot")
#     plt.savefig(os.path.join("reports", "figures", "fcModel.png"))
#
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
#     main(data_filepath, in_size, out_size, hidd_layers, dropout_rate)
