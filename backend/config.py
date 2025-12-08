# config.py
import os

# Folder dataset teks untuk melatih intent/scope/time/etc.
DATA_DIR   = os.getenv("CASE3_DATA_DIR", "/workspace/next")

# Root output semua head setelah training
OUT_ROOT   = os.getenv("CASE3_OUT_ROOT", "./runs_forecast_heads_v1")

# Base LM backbone
BASE_MODEL = os.getenv("CASE3_BASE_MODEL", "indobenchmark/indobert-base-p1")

# Hyperparams default training head-classifier/regressor
EPOCHS  = int(os.getenv("CASE3_EPOCHS", 20))
LR      = float(os.getenv("CASE3_LR", 3e-5))
BATCH   = int(os.getenv("CASE3_BATCH", 16))
MAX_LEN = int(os.getenv("CASE3_MAX_LEN", 128))
SEED    = int(os.getenv("CASE3_SEED", 42))
