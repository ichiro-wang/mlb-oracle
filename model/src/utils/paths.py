from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERMEDIATE = PROJECT_ROOT / "data" / "intermediate"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
