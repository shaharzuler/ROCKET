import os

import pandas as pd


def load_data(data_path: str):
    assert os.path.exists(data_path)
    df = pd.read_table(data_path, header=None)
    return df
