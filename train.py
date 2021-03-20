import os
import pickle

import pytorch_lightning as pl

from model import RocketNet
from utils import load_data

N_CLASSES = 2
KERNEL_COUNT = 100
TRAIN_DATA_PATH = os.path.join("data", "FordA_TRAIN.csv")
TEST_DATA_PATH = os.path.join("data", "FordA_TEST.csv")
TRAINED_MODEL_PATH = os.path.join("trained_models")
DATALOADERS_PATH = os.path.join("dataloaders")


def make_dataloaders():
    #TEMP. replace with dataloaders
    pass


def save_dataloaders():
    os.makedirs(DATALOADERS_PATH, exist_ok=True)
    train_dl_path = os.path.join(DATALOADERS_PATH, "train_dl.pkl")
    test_dl_path = os.path.join(DATALOADERS_PATH, "test_dl.pkl")
    with open(train_dl_path, "wb") as fp:
        pickle.dump(train_dataloader, fp)
    with open(test_dl_path, "wb") as fp:
        pickle.dump(val_dataloader, fp)


if __name__ == '__main__':
    df_train = load_data(TRAIN_DATA_PATH)
    df_test = load_data(TEST_DATA_PATH)

    train_dataloader, val_dataloader = make_dataloaders()

    save_dataloaders()

    model = RocketNet(x_dim=20,
                      n_classes=N_CLASSES,
                      kernel_count=KERNEL_COUNT,
                      max_sequence_len=100,
                      kernel_lengths=[7, 9, 11])


    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=TRAINED_MODEL_PATH,
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(gpus=1, checkpoint_callback=checkpoint_cb)
    trainer.fit(model, train_dataloader=None, val_dataloaders=None)