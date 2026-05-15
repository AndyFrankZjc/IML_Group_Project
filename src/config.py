from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

APPLICATION_TRAIN_PATH = DATA_DIR / "application_train.csv"
BUREAU_PATH = DATA_DIR / "bureau.csv"
PREVIOUS_APPLICATION_PATH = DATA_DIR / "previous_application.csv"
INSTALLMENTS_PAYMENTS_PATH = DATA_DIR / "installments_payments.csv"

RANDOM_STATE = 42

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
