from torch.utils.data import DataLoader

from FordADataset import FordADataset


class DatasetHandler:
    def __init__(self, df_train, df_test, batch_size):
        self.df_train = df_train
        self.df_test = df_test
        self.batch_size = batch_size

    def load_dataset(self):
        train_dataset = FordADataset(df=self.df_train)
        val_dataset = FordADataset(df=self.df_test)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)
        return train_dataloader, val_dataloader
