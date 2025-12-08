import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent

def _norm(s: str) -> str:
    """Normalisasi nama kolom biar gampang dicocokin."""
    return (s or "").strip().lower().replace(" ", "").replace("_", "")


def _auto_detect_columns(raw: pd.DataFrame) -> dict:
    """
    Deteksi kolom machine_id, ts, value dari Excel mentah.
    Return dict {"machine_id": <colname>, "ts": <colname>, "value": <colname>}
    Raise ValueError kalau ada yang gak ketemu.
    """
    cols_norm = {c: _norm(str(c)) for c in raw.columns}
    col_map = {}

    # machine id
    for c, n in cols_norm.items():
        if n in {"tagnumber", "tagnumbe", "tag", "machine", "mesin", "unit"}:
            col_map["machine_id"] = c
            break

    # timestamp
    for c, n in cols_norm.items():
        if n in {
            "timestamp/datetime",
            "timestampdatetime",
            "timestamp",
            "datetime",
            "waktu",
            "timestamptime",
        }:
            col_map["ts"] = c
            break

    # value
    for c, n in cols_norm.items():
        if n in {"value", "nilai"}:
            col_map["value"] = c
            break

    missing = {"machine_id", "ts", "value"} - set(col_map.keys())
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}. "
            f"Kolom tersedia: {list(raw.columns)}"
        )

    return col_map


def _to_float_any(x):
    """Ubah string '12,5' jadi float 12.5, dan safe-coerce selainnya."""
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def _clean_df(raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Ambil kolom penting, parse datetime & numeric, sort, dedupe.
    Hasil akhirnya: df_raw[["machine_id","ts","value"]]
    """
    df_raw = raw[[col_map["machine_id"], col_map["ts"], col_map["value"]]].copy()
    df_raw.columns = ["machine_id", "ts", "value"]

    # parse waktu
    df_raw["ts"] = pd.to_datetime(df_raw["ts"], errors="coerce")

    # parse angka
    df_raw["value"] = df_raw["value"].apply(_to_float_any)

    # bersihin
    df_raw = (
        df_raw.dropna(subset=["ts", "value"])
        .astype({"machine_id": "string"})
        .sort_values(["machine_id", "ts"])
        .drop_duplicates(subset=["machine_id", "ts"], keep="last")
        .reset_index(drop=True)
    )

    return df_raw


def _add_lags(df_raw: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """
    Tambahkan fitur lag y_lag_1..y_lag_k per mesin.
    Return df_feat (bukan inplace df_raw).
    """
    def add_lags_one_machine(group):
        g = group.sort_values("ts").copy()
        for k in range(1, max_lag + 1):
            g[f"y_lag_{k}"] = g["value"].shift(k)
        return g

    df_feat = df_raw.groupby("machine_id", group_keys=False).apply(
        add_lags_one_machine
    )
    return df_feat


def build_df_raw(
    excel_path: str,
    sheet_name: str = "SW_3min",
    save_csv: Optional[str] = None,
    with_lag: bool = False,
    max_lag: int = 6,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load Excel mentah -> normalisasi -> bersihin -> (opsional) generate lag ->
    (opsional) simpan CSV.

    Params:
    - excel_path : path file Excel sumber
    - sheet_name : nama sheet
    - save_csv   : kalau diisi path, df_raw akan di-save ke CSV
    - with_lag   : kalau True, return df_feat juga (df dengan kolom lag)
    - max_lag    : jumlah lag kalau with_lag=True

    Return:
    - (df_raw, df_feat) kalau with_lag=True
    - (df_raw, None)    kalau with_lag=False
    """
    # 1. load excel
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 2. deteksi kolom penting
    col_map = _auto_detect_columns(raw)

    # 3. beresin data
    df_raw = _clean_df(raw, col_map)

    # 4. optional save
    if save_csv is not None:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(save_csv, index=False)

    # 5. optional lag
    df_feat = None
    if with_lag:
        df_feat = _add_lags(df_raw, max_lag=max_lag)

    return df_raw, df_feat


# OPTIONAL: biar bisa dijalankan langsung dari terminal
if __name__ == "__main__":
    EXCEL_PATH = BASE_DIR/"Machine_Value_Processed.xlsx"
    SHEET_NAME = BASE_DIR/"SW_3min"
    SAVE_CSV = BASE_DIR/"df_raw.csv"

    df_raw, df_feat = build_df_raw(
        excel_path=EXCEL_PATH,
        sheet_name=SHEET_NAME,
        save_csv=SAVE_CSV,
        with_lag=True,
        max_lag=6,
    )

    print("df_raw (head):")
    print(df_raw.head())

    if df_feat is not None:
        print("\ndf_feat with lags (head):")
        print(df_feat.head(10))

    print(f"\nSaved df_raw → {SAVE_CSV}")
    
# ==============================================================
# CARA PAKAI
# from data_loader import build_df_raw

# df_raw, _ = build_df_raw(
#     excel_path="/workspace/Machine_Value_Processed.xlsx",
#     sheet_name="SW_3min",
#     save_csv="/workspace/df_raw.csv",
#     with_lag=False,
# )
# ==============================================================

# import pandas as pd
# import numpy as np
# from pathlib import Path

# # ====== CONFIG ======
# EXCEL_PATH = "/workspace/Machine_Value_Processed.xlsx"   # ganti ke path file kamu
# SHEET_NAME = "SW_3min"                    # sheet yang kamu sebut
# SAVE_CSV   = "/workspace/df_raw.csv"     # opsional: simpan ke CSV

# # ====== LOAD ======
# # read_excel otomatis pilih engine (openpyxl/xlrd) sesuai tipe file
# raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# # ====== NORMALISASI NAMA KOLOM ======
# def norm(s: str) -> str:
#     return (s or "").strip().lower().replace(" ", "").replace("_", "")

# cols_norm = {c: norm(str(c)) for c in raw.columns}

# # Kandidat nama kolom (beberapa kemungkinan dari screenshot)
# # - machine/tag: "tagnumber", "tagnumbe" (typo), "tag", "machine", "mesin"
# # - timestamp  : "timestamp/datetime", "timestamp", "datetime", "waktu"
# # - value      : "value", "nilai"
# col_map = {}

# # cari kolom machine
# for c, n in cols_norm.items():
#     if n in {"tagnumber","tagnumbe","tag","machine","mesin","unit"}:
#         col_map["machine_id"] = c
#         break

# # cari kolom timestamp
# for c, n in cols_norm.items():
#     if n in {"timestamp/datetime","timestampdatetime","timestamp","datetime","waktu","timestamptime"}:
#         col_map["ts"] = c
#         break

# # cari kolom value
# for c, n in cols_norm.items():
#     if n in {"value","nilai"}:
#         col_map["value"] = c
#         break

# # validasi
# missing = {"machine_id","ts","value"} - set(col_map.keys())
# if missing:
#     raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. Kolom tersedia: {list(raw.columns)}")

# # ====== BENTUK DF_RAW ======
# df_raw = raw[[col_map["machine_id"], col_map["ts"], col_map["value"]]].copy()
# df_raw.columns = ["machine_id", "ts", "value"]

# # 1) parse waktu
# df_raw["ts"] = pd.to_datetime(df_raw["ts"], errors="coerce")

# # 2) angka dengan koma sebagai desimal → ganti koma jadi titik dulu
# #    (kalau sudah titik, ini tidak mengubah apa-apa)
# def to_float_any(x):
#     if isinstance(x, str):
#         x = x.replace(",", ".")
#     return pd.to_numeric(x, errors="coerce")

# df_raw["value"] = df_raw["value"].apply(to_float_any)

# # 3) bersihkan
# df_raw = df_raw.dropna(subset=["ts","value"]).astype({"machine_id": "string"})
# df_raw = df_raw.sort_values(["machine_id", "ts"]).drop_duplicates(subset=["machine_id","ts"], keep="last").reset_index(drop=True)

# print("df_raw (head):")
# print(df_raw.head())

# # ====== (OPSIONAL) FITUR TAMBAHAN ala screenshot (lag 1..6) ======
# # ini tidak wajib, tapi kalau kamu perlu kolom y_lag_1..y_lag_6 per mesin:
# def add_lags(group, max_lag=6):
#     g = group.sort_values("ts").copy()
#     for k in range(1, max_lag+1):
#         g[f"y_lag_{k}"] = g["value"].shift(k)
#     return g

# df_feat = df_raw.groupby("machine_id", group_keys=False).apply(add_lags, max_lag=6)

# print("\nDengan fitur lag (head):")
# print(df_feat.head(10))

# # ====== (OPSIONAL) SIMPAN ======
# Path(SAVE_CSV).parent.mkdir(parents=True, exist_ok=True)
# df_raw.to_csv(SAVE_CSV, index=False)
# print(f"\nSaved df_raw → {SAVE_CSV}")
