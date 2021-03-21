import torch
from torch.utils.data import Dataset


class FordADataset(Dataset):
    """Dataset which samples the data from hourly electricity data. source: class material, InceptionTime.ipynb"""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        class_label = int(sample.iloc[0] == 1)
        values = sample.iloc[1:].values

        return (torch.Tensor(values.reshape(1, -1)),
                torch.Tensor([class_label]))
