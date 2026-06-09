"""
loader.py

pull statcast data from the internet and filter plate appearances

Usage:
python -m src.data.loader
"""

from pybaseball import statcast
import pandas as pd
from datetime import date
from src.utils.constants import PA_COLS
from src.utils.paths import DATA_RAW, DATA_INTERMEDIATE

PA_FILE = DATA_INTERMEDIATE / "pa.parquet"
SEASONS = [2024, 2025, 2026]


def load_statcast(seasons: list[tuple[str, str]]) -> pd.DataFrame:
    """Pull and concatenate multiple seasons of Statcast data."""
    frames = []
    for start, end in seasons:
        print(f"Pulling {start} → {end} ...")
        df = statcast(start_dt=start, end_dt=end)
        df = df[df["game_type"] == "R"]
        cache_name = DATA_RAW / f"statcast_{start[:4]}.parquet"
        df.to_parquet(cache_name)
        print(f"  Saved {cache_name} ({len(df):,} pitches)")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last pitch of each plate appearance."""
    df = df[df["events"].notna()][PA_COLS].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] <= pd.Timestamp(date.today())]  # no future rows
    return df.sort_values("game_date").reset_index(drop=True)


def load_raw_data(fetch_new: bool = False) -> pd.DataFrame:
    """
    Loading Statcast data from internet or local cache, depending on `fetch_new`
    """
    if fetch_new:
        # Fetch from internet
        years = [(f"{yr}-01-01", f"{yr}-12-31") for yr in SEASONS]
        raw = load_statcast(years)
    else:
        # Fetch from local cache
        years = [pd.read_parquet(DATA_RAW / f"statcast_{yr}.parquet") for yr in SEASONS]
        raw = pd.concat(years, ignore_index=True)

    return raw


def load_pa_data(fetch_new: bool = False, rebuild: bool = False) -> pd.DataFrame:
    """
    Return PA-level dataset.

    Priority:
    1. fetch_new=True -> rebuild from Statcast
    2. existing pa.parquet -> load cached file
    3. build pa.parquet from raw season files
    """

    if fetch_new:
        print("Building PA data from online statcast data...")
        raw = load_raw_data(fetch_new=True)
        df = filter_to_pa(raw)
        df.to_parquet(PA_FILE, index=False)
        return df

    if PA_FILE.exists() and not rebuild:
        print("Loading cached PA data...")
        df = pd.read_parquet(PA_FILE)
        return df

    print("Building PA data from raw season files...")
    raw = load_raw_data(fetch_new=False)
    df = filter_to_pa(raw)
    df.to_parquet(PA_FILE, index=False)

    return df


def main():
    load_pa_data(fetch_new=False)


if __name__ == "__main__":
    main()
