"""
features.py

adding features to dataset
Usage:
python -m src.data.features
"""

import pandas as pd
from src.utils.constants import PARK_FACTORS_HR, OUTCOME_MAP, OUTCOMES
from src.utils.paths import DATA_PROCESSED
from src.data.loader import load_pa_data

GROUPS = ["batter", "pitcher"]


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """sort the PA's by date once at the start"""
    df = df.sort_values("game_date")
    return df


def add_park_features(df: pd.DataFrame) -> pd.DataFrame:
    """adding park factors such as HR factor"""
    df["park_factors_hr"] = df["home_team"].map(PARK_FACTORS_HR).fillna(1.0)
    return df


def add_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    """one hot encode for outcomes"""
    df["outcome"] = df["events"].map(OUTCOME_MAP).fillna("OUT")
    for outcome in OUTCOMES:
        df[f"is_{outcome}"] = (df["outcome"] == outcome).astype(int)
    return df


def remove_truncated_pa(df: pd.DataFrame) -> pd.DataFrame:
    """remove truncated/incomplete PA's"""
    df = df[df["outcome"] != "TRC"]
    return df


def rolling_rate(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    window: int = 100,
    min_pa: int = 20,
) -> pd.Series:
    """
    Rolling PA-count window, shifted by 1 to avoid data leakage.
    """
    return df.groupby(group_col)[outcome_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=min_pa).mean()
    )


def add_rolling_rates(
    df: pd.DataFrame, window: int = 100, min_pa: int = 20
) -> pd.DataFrame:
    for group in GROUPS:
        for outcome in OUTCOMES:
            col = f"is_{outcome}"
            df[f"{group}_{window}pa_{outcome}_rate"] = rolling_rate(
                df, group, col, window, min_pa
            )
    return df


def impute_rolling_rates(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    rate_cols = [c for c in df.columns if f"_{window}pa_" in c and "_rate" in c]
    league_means = df[rate_cols].mean()
    df[rate_cols] = df[rate_cols].fillna(league_means)
    return df


def add_imputation_flags(df, window: int = 100):
    for group in GROUPS:
        rate_cols = [c for c in df.columns if f"{group}_{window}pa_" in c]
        df[f"{group}_imputed"] = df[rate_cols].isnull().any(axis=1).astype(int)
    return df


def add_rolling_xwoba(
    df: pd.DataFrame, window: int = 100, min_pa: int = 20
) -> pd.DataFrame:
    for group in GROUPS:
        df[f"{group}_{window}pa_xwoba"] = df.groupby(group)[
            "estimated_woba_using_speedangle"
        ].transform(lambda x: (x.shift(1).rolling(window, min_periods=min_pa).mean()))

    return df


def add_rolling_platoon_xwoba(
    df: pd.DataFrame, window: int = 100, min_pa: int = 20
) -> pd.DataFrame:
    for group in GROUPS:
        df[f"{group}_{window}pa_platoon_xwoba"] = df.groupby(
            [group, "stand", "p_throws"]
        )["estimated_woba_using_speedangle"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min_pa).mean()
        )

        df[f"{group}_platoon_pa"] = df.groupby([group, "stand", "p_throws"]).cumcount()
    return df


def add_game_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df["runners_on_1b"] = df["on_1b"].notna().astype(int)
    df["runners_on_2b"] = df["on_2b"].notna().astype(int)
    df["runners_on_3b"] = df["on_3b"].notna().astype(int)
    df["score_diff"] = df["bat_score"] - df["fld_score"]
    df["is_bottom"] = (df["inning_topbot"] == "Bot").astype(int)
    return df


def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_remove = [
        "on_1b",
        "on_2b",
        "on_3b",
        "inning_topbot",
        "events",
        "outcome",
        "game_pk",
        "at_bat_number",
        "home_team",
        "away_team",
        "bat_score",
        "fld_score",
        "estimated_woba_using_speedangle",
        "launch_speed",
        "balls",
        "strike",
    ]
    df = df.drop(columns=[c for c in to_remove if c in df.columns])
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = sort_by_date(df)
    df = add_park_features(df)

    # remove truncated pa after adding outcome features
    df = add_outcome_features(df)
    df = remove_truncated_pa(df)

    window = 100  # 100pa for now
    df = add_rolling_rates(df, window=window)
    df = add_rolling_xwoba(df, window=window)
    df = add_rolling_platoon_xwoba(df, window=window)
    df = add_game_state_features(df)

    # flag before impute
    df = add_imputation_flags(df, window=window)
    df = impute_rolling_rates(df, window=window)

    df = remove_columns(df)
    return df


def main():
    df = load_pa_data()
    df = build_dataset(df)
    print("Saving processed PA data")
    df.to_parquet(DATA_PROCESSED / "pa_features.parquet", index=False)


if __name__ == "__main__":
    main()
