import os
import pandas as pd
import pytorch_lightning as pl

from model import RocketNet
from utils import load_data

N_CLASSES = 2
KERNEL_COUNT = 100
TRAIN_DATA_PATH = os.path.join("data", "FordA_TRAIN.csv")
TEST_DATA_PATH = os.path.join("data", "FordA_TEST.csv")
TRAINED_MODEL_PATH = os.path.join("trained_models")


if __name__ == '__main__':
    df_train = load_data(TRAIN_DATA_PATH)
    df_test = load_data(TEST_DATA_PATH)

    model = RocketNet(
        n_classes=N_CLASSES,
        kernel_count=KERNEL_COUNT,
        kernel_config=None
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=TRAINED_MODEL_PATH,
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(gpus=1, checkpoint_callback=checkpoint_cb)
    trainer.fit(model, train_dataloader=None, val_dataloaders=None)