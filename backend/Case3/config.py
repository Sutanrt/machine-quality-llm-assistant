# config.py
import os

# Folder dataset teks untuk melatih intent/scope/time/etc.
from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR   = os.getenv("CASE3_DATA_DIR", BASE_DIR/"next")

# Root output semua head setelah training
OUT_ROOT   = os.getenv("CASE3_OUT_ROOT", BASE_DIR/"Case3/runs_forecast_heads_v1")

# Base LM backbone
BASE_MODEL = os.getenv("CASE3_BASE_MODEL", "indobenchmark/indobert-base-p1")

# Hyperparams default training head-classifier/regressor
EPOCHS  = int(os.getenv("CASE3_EPOCHS", 20))
LR      = float(os.getenv("CASE3_LR", 3e-5))
BATCH   = int(os.getenv("CASE3_BATCH", 16))
MAX_LEN = int(os.getenv("CASE3_MAX_LEN", 128))
SEED    = int(os.getenv("CASE3_SEED", 42))


# # config.py
# BASE_MODEL = "indobenchmark/indobert-base-p1"

# # lokasi training head NLU yang sudah kamu simpan per head:
# OUT_ROOT = "/workspace/runs_forecast_heads_v1"

# lokasi model classical per horizon (linear + RF residual)

FORECASTER_ROOT = BASE_DIR/"outputs_horizon_models"

# lokasi dataset time series mentah
DATA_CSV = BASE_DIR/"df_raw2.csv"
