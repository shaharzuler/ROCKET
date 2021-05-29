import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import RocketNet

CACHED_MODEL = os.path.join("trained_models", "cached_model.pkl")
TRAIN_DL_PATH = os.path.join("dataloaders", "train_dl.pkl")
TEST_DL_PATH = os.path.join("dataloaders", "test_dl.pkl")


def plot_ts(x_series: np.array, y: int, pred: int):
    color = "green" if y > 0 else "red"
    plt.Figure()
    if len(x_series.shape) > 1:
        x_series = x_series.squeeze()
    plt.plot(x_series, c=color)
    plt.grid()
    plt.title(f"Real label: {int(y.item())}, Predicted label: {pred}")
    plt.show()


def predict(model: RocketNet, dataset: DataLoader, index: int):
    data_sample = dataset[index]
    x = data_sample[0].unsqueeze(dim=0)
    y_true = data_sample[1]
    with torch.no_grad():
        prob = model(x).item()
        pred = 1 if prob >= model.thr else 0
        print(f"model proba: {prob}, model prediction: {pred}, real label: {y_true}")
        plot_ts(x, y_true, pred)
    return prob, pred, y_true


if __name__ == '__main__':
    if os.path.exists(CACHED_MODEL):
        with open(CACHED_MODEL, "rb") as fp:
            model: RocketNet = pickle.load(fp)
    else:
        # get model with minimal loss
        trained_model_list = os.listdir("trained_models")
        best_model_index = np.argmin([int(name.split("=")[2].split(".")[0]) for name in trained_model_list])
        TRAINED_MODEL_NAME = os.listdir("trained_models")[best_model_index]
        TRAINED_MODEL_PATH = os.path.join("trained_models", TRAINED_MODEL_NAME)
        print(f"loaded model: {TRAINED_MODEL_PATH}")
        model = RocketNet.load_from_checkpoint(TRAINED_MODEL_PATH)
        with open(CACHED_MODEL, "wb") as fp:
            pickle.dump(model, fp)
    model.eval()

    # test data
    with open(TEST_DL_PATH, "rb") as fp:
        pred_dl = pickle.load(fp)

    pred_dataset = pred_dl.dataset
    index = random.randint(0, len(pred_dataset))
    print(f"sampled index: {index}")

    predict(model, pred_dataset, index)
