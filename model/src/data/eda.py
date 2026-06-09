"""
eda.py
Sanity checks and exploratory analysis on the PA feature dataset.
Run after features.py has produced pa_features.parquet.

Usage:
    python -m src.data.eda
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.utils.constants import OUTCOMES
from src.utils.paths import DATA_PROCESSED, REPORTS


# ── 1. Load ────────────────────────────────────────────────────────────────────


def load(path=DATA_PROCESSED / "pa_features.parquet") -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows  |  {df.shape[1]} columns")
    return df


# ── 2. Basic shape & dtypes ────────────────────────────────────────────────────


def check_shape(df: pd.DataFrame) -> None:
    print("\n── Shape & dtypes ──")
    print(df.dtypes.value_counts().to_string())
    print(
        f"\nDate range: {df['game_date'].min().date()} → {df['game_date'].max().date()}"
    )
    print(f"Unique games:   {df['game_pk'].nunique():,}")
    print(f"Unique batters: {df['batter'].nunique():,}")
    print(f"Unique pitchers:{df['pitcher'].nunique():,}")


# ── 3. Missing values ──────────────────────────────────────────────────────────


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Missing values ──")
    miss = (
        df.isnull()
        .sum()
        .rename("n_missing")
        .to_frame()
        .assign(pct=lambda x: x["n_missing"] / len(df) * 100)
        .query("n_missing > 0")
        .sort_values("pct", ascending=False)
    )
    if miss.empty:
        print("  No missing values.")
    else:
        print(miss.to_string())

    # Rolling-rate coverage (expected to have NaNs for early PAs)
    rate_cols = [c for c in df.columns if "_100pa_" in c or "_platoon_" in c]
    if rate_cols:
        print("\n── Rolling feature NaN coverage ──")
        cov = (df[rate_cols].isnull().mean() * 100).sort_values(ascending=False)
        print(cov.to_string())
        print(
            f"\n  Rows where ALL rolling features are NaN: "
            f"{df[rate_cols].isnull().all(axis=1).sum():,} "
            f"({df[rate_cols].isnull().all(axis=1).mean() * 100:.1f}%)"
        )
    return miss


# ── 4. Outcome distribution ────────────────────────────────────────────────────


def check_outcomes(df: pd.DataFrame) -> None:
    print("\n── Outcome distribution ──")

    # Raw event counts (unmapped)
    raw_counts = df["events"].value_counts()
    print("\nRaw statcast events:")
    print(raw_counts.to_string())

    # Mapped outcome counts
    mapped_counts = df["outcome"].value_counts()
    print("\nMapped outcomes:")
    print(mapped_counts.to_string())

    # Check events that map to OUT — are any surprising?
    out_events = df.loc[df["outcome"] == "OUT", "events"].value_counts()
    print("\nEvents mapped → OUT:")
    print(out_events.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mapped_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Outcome distribution (mapped)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    pct = mapped_counts / mapped_counts.sum() * 100
    pct.plot(kind="bar", ax=axes[1], color="darkorange", edgecolor="white")
    axes[1].set_title("Outcome distribution (% of PAs)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = REPORTS / "outcome_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved → {out}")


# ── 5. Class imbalance summary ─────────────────────────────────────────────────


def check_imbalance(df: pd.DataFrame) -> None:
    print("\n── Class imbalance ──")
    total = len(df)
    rows = []
    for o in OUTCOMES:
        col = f"is_{o}"
        if col in df.columns:
            n = df[col].sum()
            rows.append({"outcome": o, "count": n, "pct": n / total * 100})
    imb = pd.DataFrame(rows).sort_values("pct", ascending=False)
    print(imb.to_string(index=False))
    minority = imb[imb["pct"] < 5]
    if not minority.empty:
        print(
            f"\n  ⚠  {len(minority)} outcomes have <5% frequency — "
            "consider weighted loss or oversampling."
        )


# ── 6. Leakage spot-check ──────────────────────────────────────────────────────


def check_leakage(df: pd.DataFrame, n_sample: int = 3) -> None:
    """
    For a few batters, verify that the rolling rate at row T
    does not reflect the outcome of row T itself.
    """
    print("\n── Leakage spot-check ──")
    rate_cols = [c for c in df.columns if "batter_100pa_" in c]
    if not rate_cols:
        print("  No batter rolling-rate columns found.")
        return

    sample_col = rate_cols[0]
    outcome_col = sample_col.replace("batter_100pa_", "is_").replace("_rate", "")

    for batter_id in df["batter"].value_counts().head(n_sample).index:
        sub = (
            df[df["batter"] == batter_id][["game_date", outcome_col, sample_col]]
            .dropna()
            .head(10)
        )
        print(f"\n  Batter {batter_id} — {outcome_col} vs {sample_col}:")
        print(sub.to_string(index=False))


# ── 7. Rolling feature distributions ──────────────────────────────────────────


def plot_rolling_features(df: pd.DataFrame) -> None:
    rate_cols = [c for c in df.columns if "_100pa_" in c]
    if not rate_cols:
        return

    n = len(rate_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(rate_cols):
        df[col].dropna().hist(bins=40, ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=7)
        axes[i].tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Rolling rate feature distributions", y=1.01)
    plt.tight_layout()
    out = REPORTS / "rolling_feature_distributions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")


# ── 8. Season-level summary ────────────────────────────────────────────────────


def check_seasons(df: pd.DataFrame) -> None:
    print("\n── PAs per season ──")
    summary = (
        df.assign(season=df["game_date"].dt.year)
        .groupby("season")
        .agg(n_pa=("batter", "count"), n_games=("game_pk", "nunique"))
    )
    print(summary.to_string())


# ── 9. Inspect all columns ────────────────────────────────────────────────────


def show_columns(df):
    print("\nNumeric:")
    print(df.select_dtypes(include="number").columns.tolist())

    print("\nCategorical:")
    print(df.select_dtypes(include="object").columns.tolist())

    print("\nDatetime:")
    print(df.select_dtypes(include="datetime").columns.tolist())


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    df = load()
    # check_shape(df)
    # check_missing(df)
    # check_outcomes(df)
    # check_imbalance(df)
    # check_leakage(df)
    # plot_rolling_features(df)
    # check_seasons(df)
    show_columns(df)
    print("\n✓ EDA complete. Check reports/ for saved plots.")


if __name__ == "__main__":
    main()
