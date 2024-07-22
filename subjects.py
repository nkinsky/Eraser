import pandas as pd
import numpy as np
from pathlib import Path

group_data_dir = Path("/Users/nkinsky/Documents/BU/Working/Eraser")


def save_df(df: pd.DataFrame, savename: str, save_dir=group_data_dir):
    """data = pd.DataFrame typically"""
    assert isinstance(df, pd.DataFrame)

    savename = savename if savename.split(".")[-1] == "csv" else f"{savename}.csv"
    df.to_csv(save_dir / savename, index=False)
    print(f"{savename} saved")


def load_df(savename: str, save_dir=group_data_dir, header='infer'):
    savename = savename if savename.split(".")[-1] == "csv" else f"{savename}.csv"
    return pd.read_csv(save_dir / savename, header=header)