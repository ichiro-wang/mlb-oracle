from pybaseball import statcast
import pandas as pd
import numpy as np
from utils.paths import DATA_PROCESSED, DATA_RAW

PARK_HR_FACTORS = {
    "COL": 1.30,
    "CIN": 1.15,
    "TEX": 1.12,
    "PHI": 1.10,
    "BAL": 1.08,
    "BOS": 1.05,
    "NYY": 1.04,
    "MIL": 1.03,
    "ATL": 1.02,
    "HOU": 1.01,
    "ARI": 1.00,
    "LAA": 0.99,
    "DET": 0.98,
    "MIN": 0.97,
    "CLE": 0.96,
    "WSH": 0.95,
    "CHC": 0.95,
    "TOR": 0.94,
    "STL": 0.94,
    "KC": 0.93,
    "NYM": 0.93,
    "PIT": 0.92,
    "MIA": 0.92,
    "TB": 0.91,
    "LAD": 0.91,
    "CHW": 0.90,
    "SD": 0.90,
    "SEA": 0.89,
    "OAK": 0.88,
    "SF": 0.77,
}

PA_COLS = [
    "game_date",
    "batter",
    "pitcher",
    "stand",
    "p_throws",
    "events",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "balls",
    "strikes",
    "bat_score",
    "fld_score",
    "launch_speed",
    "estimated_woba_using_speedangle",
]

OUTCOMES = ["1B", "2B", "3B", "HR", "BB", "SO", "HBP", "GDP", "SF", "SH"]

OUTCOME_MAP = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "walk": "BB",
    "strikeout": "SO",
    "hit_by_pitch": "HBP",
    "grounded_into_double_play": "GDP",
    "sac_fly": "SF",
    "sac_bunt": "SH",
}


def load_statcast(seasons: list[tuple[str, str]]) -> pd.DataFrame:
    """Pull and concatenate multiple seasons of Statcast data."""
    frames = []
    for start, end in seasons:
        print(f"Pulling {start} → {end} ...")
        df = statcast(start_dt=start, end_dt=end)
        cache_name = DATA_RAW / f"statcast_{start[:4]}.parquet"
        df.to_parquet(cache_name)
        print(f"  Saved {cache_name} ({len(df):,} pitches)")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last pitch of each plate appearance."""
    pa = df[df["events"].notna()][PA_COLS].copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    return pa.sort_values("game_date").reset_index(drop=True)


def add_park_features(pa: pd.DataFrame) -> pd.DataFrame:
    pa["park_hr_factor"] = pa["home_team"].map(PARK_HR_FACTORS).fillna(1.0)
    return pa


def add_outcome_features(pa: pd.DataFrame) -> pd.DataFrame:
    pa["outcome"] = pa["events"].map(OUTCOME_MAP).fillna("OUT")
    for outcome in OUTCOMES:
        pa[f"is_{outcome}"] = (pa["outcome"] == outcome).astype(int)
    return pa


def rolling_rate(
    df: pd.DataFrame, group_col, outcome_col: str, window: int = 500, min_pa: int = 20
) -> pd.Series:
    """
    Rolling PA-count window, shifted by 1 to avoid data leakage.
    window=500 ≈ one full season of plate appearances.
    """
    return df.groupby(group_col)[outcome_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=min_pa).mean()
    )


def add_rolling_rates(pa: pd.DataFrame) -> pd.DataFrame:
    for outcome in OUTCOMES:
        col = f"is_{outcome}"
        pa[f"batter_365_{outcome}_rate"] = rolling_rate(pa, "batter", col)
        pa[f"pitcher_365_{outcome}_rate"] = rolling_rate(pa, "pitcher", col)
    return pa


def add_platoon_features(pa: pd.DataFrame) -> pd.DataFrame:
    pa["matchup"] = pa["stand"] + "v" + pa["p_throws"]  # LvL, LvR, RvL, RvR
    pa["batter_platoon_woba"] = pa.groupby(["batter", "matchup"])[
        "estimated_woba_using_speedangle"
    ].transform(lambda x: x.shift(1).rolling(200, min_periods=20).mean())
    pa["pitcher_platoon_woba"] = pa.groupby(["pitcher", "matchup"])[
        "estimated_woba_using_speedangle"
    ].transform(lambda x: x.shift(1).rolling(200, min_periods=20).mean())
    return pa


def add_game_state_features(pa: pd.DataFrame) -> pd.DataFrame:
    pa["runners_on_1b"] = pa["on_1b"].notna().astype(int)
    pa["runners_on_2b"] = pa["on_2b"].notna().astype(int)
    pa["runners_on_3b"] = pa["on_3b"].notna().astype(int)
    pa["score_diff"] = pa["bat_score"] - pa["fld_score"]
    pa["is_bottom"] = (pa["inning_topbot"] == "Bot").astype(int)
    return pa


def build_features(pa: pd.DataFrame) -> pd.DataFrame:
    pa = add_park_features(pa)
    pa = add_outcome_features(pa)
    pa = add_rolling_rates(pa)
    pa = add_platoon_features(pa)
    pa = add_game_state_features(pa)
    return pa


def main():
    # Pull 2024 + 2025
    raw = load_statcast(
        [
            ("2024-01-01", "2024-12-31"),
            ("2025-01-01", "2025-12-31"),
        ]
    )

    # --- After first run, load from cache instead: ---
    # raw = pd.concat(
    #     [
    #         pd.read_parquet("statcast_2024.parquet"),
    #         pd.read_parquet("statcast_2025.parquet"),
    #     ]
    # )

    pa = filter_to_pa(raw)
    print(f"\n{len(pa):,} plate appearances loaded")

    pa = build_features(pa)

    feature_cols = [
        c for c in pa.columns if "batter_365" in c or "pitcher_365" in c
    ] + [
        "batter_platoon_woba",
        "pitcher_platoon_woba",
        "park_hr_factor",
        "score_diff",
        "inning",
        "outs_when_up",
        "runners_on_1b",
        "runners_on_2b",
        "runners_on_3b",
        "is_bottom",
        "balls",
        "strikes",
    ]

    print("\nNull rates per feature:")
    print(pa[feature_cols].isnull().mean().sort_values(ascending=False).to_string())

    pa.to_parquet(DATA_PROCESSED / "pa_features.parquet")
    print("\nSaved pa_features.parquet")


if __name__ == "__main__":
    main()
