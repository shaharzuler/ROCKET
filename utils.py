import os
import pandas as pd


def load_data(data_path: str):
    assert os.path.exists(data_path)
    col_names = [f"x_{i}" for i in range(500)]
    col_names.insert(0, "label")
    df = pd.read_csv(data_path, sep="\t", names=col_names)
    return df
