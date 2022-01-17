# # -*- coding: utf-8 -*-
# import argparse
# import logging
# import os.path
# from pathlib import Path
#
# import torch
# from torch.utils.data import DataLoader, TensorDataset
#
# parser = argparse.ArgumentParser(description="T-SNE visualization arguments")
# parser.add_argument("--data_path", type=Path, default=os.path.join("data", "processed"))
# parser.add_argument(
#     "--model_path", type=Path, default=os.path.join("models", "fcModel.pt")
# )
#
# args = parser.parse_args()
# data_path = args.data_path
# model_path = args.model_path
#
# #
# # def get_activation(name):
# #     def hook(model, input, output):
# #         activation[name] = output.detach()
# #
# #     return hook
#
# #
# # def main(model_path, data_path):
# #     """extracts some intermediate representation of the data"""
# #     logger = logging.getLogger(__name__)
# #     logger.info("Loading the pretrained model")
# #     pdb.set_trace()
# #     model = torch.load(model_path)
# #     model.output.register_forward_hook(get_activation("output"))
# #     logger.info("Loading the data")
# #     images = torch.load(os.path.join(data_path, "train_images.pt"))
# #     labels = torch.load(os.path.join(data_path, "train_labels.pt"))
# #     train_set = DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True)
# #
# #     features = np.empty(0)
# #     for images, labels in train_set:
# #         output = model(images)
# #         current_outputs = output.numpy()
# #         features = np.concatenate((features, current_outputs))
# #     pdb.set_trace()
# #     tsne = TSNE(n_components=2).fit_transform(features)
#
# # logger.info("Loading the data")
# # images = np.load(data_filepath)
# # images = torch.Tensor(np.array(images))
# # images = images.view(images.shape[0], -1)
#
# #
# # images = images.view(images.shape[0], -1)
# # logger.info('Predicting the digit')
# # logits = model(images)
# # ps = torch.exp(logits)
# # top_p, top_class = ps.topk(1, dim=1)
# # print(top_class)
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
#     main(model_path, data_path)
