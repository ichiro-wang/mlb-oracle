"""
split.py

Split the data into train / val / test: 60 / 20 / 20

Usage:
python -m src.data.split
"""

import pandas as pd


def split_by_date(df: pd.DataFrame, train_pct: float = 0.6, val_pct: float = 0.2):
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_pct)]
    val_end = dates[train_end + int(n * val_pct)]

    train = df[df["game_date"] < train_end]
    val = df[df["game_date"] >= train_end & df["game_date"] < val_end]
    test = df[df["game_date"] >= val_end]

    return train, val, test
