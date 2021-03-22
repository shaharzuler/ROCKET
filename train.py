import os
import pickle

import pytorch_lightning as pl

from DatasetHandler import DatasetHandler
from model import RocketNet
from utils import load_data

N_CLASSES = 1
KERNEL_COUNT = 10000
TRAIN_DATA_PATH = os.path.join("data", "FordA_TRAIN.tsv")
TEST_DATA_PATH = os.path.join("data", "FordA_TEST.tsv")
TRAINED_MODEL_PATH = os.path.join("trained_models")
DATALOADERS_PATH = os.path.join("dataloaders")


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
    kernel_seed = 42
    batch_size = 256
    max_sequence_len = 500
    train_dataloader, val_dataloader = DatasetHandler(df_train, df_test, batch_size).load_dataset()

    save_dataloaders()

    model = RocketNet(x_dim=1,
                      n_classes=N_CLASSES,
                      kernel_seed=kernel_seed,
                      kernel_count=KERNEL_COUNT,
                      max_sequence_len=max_sequence_len)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=TRAINED_MODEL_PATH,
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(gpus=0, checkpoint_callback=checkpoint_cb)
    trainer.fit(model, train_dataloader, val_dataloader)
