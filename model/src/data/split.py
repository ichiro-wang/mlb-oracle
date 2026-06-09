"""
split.py

Split the data into train / val / test: 60 / 20 / 20

Usage:
python -m src.data.split
"""

import pandas as pd


def split_by_date(
    df: pd.DataFrame,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
):
    dates = sorted(df["game_date"].unique())

    train_idx = int(len(dates) * train_pct)
    val_idx = int(len(dates) * (train_pct + val_pct))

    train_end = dates[train_idx]
    val_end = dates[val_idx]

    train = df[df["game_date"] < train_end]
    val = df[(df["game_date"] >= train_end) & (df["game_date"] < val_end)]
    test = df[df["game_date"] >= val_end]

    print(
        f"\nSplit sizes — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )

    print(
        f"Date ranges — "
        f"train: {train['game_date'].min().date()} → {train['game_date'].max().date()} | "
        f"val: {val['game_date'].min().date()} → {val['game_date'].max().date()} | "
        f"test: {test['game_date'].min().date()} → {test['game_date'].max().date()}"
    )

    return train, val, test
