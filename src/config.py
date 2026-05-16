from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PROCESSED_PATH = DATA_DIR / "application_train_processed.csv"
TEST_PROCESSED_PATH = DATA_DIR / "application_test_processed.csv"

TRAIN_MERGED_PATH = DATA_DIR / "application_train_processed_merge_features.csv"
TEST_MERGED_PATH = DATA_DIR / "application_test_processed_merge_features.csv"

RANDOM_STATE = 42
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
