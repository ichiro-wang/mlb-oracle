"""
pipeline.py

Orchestrator for running loader.py and features.py

Usage:
python -m src.data.pipeline
"""

from src.data.loader import load_pa_data
from src.data.features import build_dataset
from src.utils.paths import DATA_PROCESSED


def main():
    df = load_pa_data(fetch_new=False, rebuild=True)
    df = build_dataset(df)
    df.to_parquet(DATA_PROCESSED / "pa_features.parquet", index=False)
    print(f"Done. {len(df):,} rows saved.")


if __name__ == "__main__":
    main()
