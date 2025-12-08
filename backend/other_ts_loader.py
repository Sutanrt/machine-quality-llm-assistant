# ============================================================
# other_ts_loader.py
# Loader untuk df_raw2.csv (post-ETL) → dataframe timeseries standar
#
# Output kolom:
#   machine_id (UPPERCASE str)
#   ts         (datetime64[ns], parsed)
#   value      (float)
#
# Cocok dipakai langsung oleh anomaly pipeline.
# ============================================================

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd


# ---------------------------------
# Helper normalisasi nama kolom
# ---------------------------------

def _norm_colname(name: str) -> str:
    """
    Normalisasi nama kolom biar bisa dicocokkan dengan fleksibel.
    Contoh:
      "Time Stamp / Date" -> "timestampdate"
      "TAG Number"        -> "tagnumber"
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def _build_col_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Deteksi kolom mana yang jadi:
      - machine_id
      - ts
      - value

    Menggunakan pendekatan fuzzy berdasarkan nama kolom.
    """
    col_map: Dict[str, str] = {}

    for c in df.columns:
        norm_c = _norm_colname(c)

        # kandidat machine
        if norm_c in {
            "machineid", "machine", "mesin", "unit", "tag", "tagnumber", "tagnumbe",
        }:
            col_map[c] = "machine_id"

        # kandidat timestamp
        elif norm_c in {
            "ts", "timestamp", "datetime", "timestamptime",
            "timestampdatetime", "datetimestamp",
        }:
            col_map[c] = "ts"

        # kandidat value
        elif (
            norm_c in {"value", "nilai", "val", "pv", "reading", "processvalue"}
            or "value" in norm_c
        ):
            col_map[c] = "value"

        # kalau file udah rapi sebelumnya
        elif c in {"machine_id", "ts", "value"}:
            col_map[c] = c

    return col_map


# ---------------------------------
# Debug helper (opsional)
# ---------------------------------

def debug_ts_coverage(df_ts: pd.DataFrame, *, max_print: int = 50, verbose: bool = False) -> None:
    """
    Cetak ringkasan distribusi timestamp kalau verbose=True.
    Ini bantu ngecek apakah parsing waktu udah bener.
    Default verbose=False supaya aman di production.
    """
    if not verbose:
        return

    print("=== DEBUG TS COVERAGE ===")
    print("dtypes :", df_ts.dtypes.to_dict())
    print("ts.min:", df_ts["ts"].min(), "ts.max:", df_ts["ts"].max())
    print()

    vc_date = (
        df_ts["ts"]
        .dt.date
        .value_counts()
        .sort_index()
    )
    print("Jumlah baris per tanggal:")
    print(vc_date.head(max_print).to_string())
    print("unique days:", vc_date.index.nunique())
    print()

    vc_month = (
        df_ts["ts"]
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )
    print("Jumlah baris per bulan:")
    print(vc_month.to_string())

    print("\nDistribusi nomor-hari (1..31) dalam df_ts:")
    vc_daynum = df_ts["ts"].dt.day.value_counts().sort_index()
    print(vc_daynum.to_string())
    print("=========================\n")


# ---------------------------------
# Loader utama
# ---------------------------------

def load_timeseries_from_df_raw2(
    csv_path: str,
    *,
    assume_us_datetime: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Baca df_raw2.csv (atau file serupa hasil ETL) jadi dataframe siap pakai:
      machine_id | ts | value

    Parameter:
    - assume_us_datetime:
        True  -> parse ts sebagai month-first (MM/DD/YYYY HH:MM[:SS])
                -> dayfirst=False
        False -> parse ts sebagai day-first (DD/MM/YYYY HH:MM[:SS])
                -> dayfirst=True
      Pilihan ini penting karena kadang data PLC/historian export formatnya US.

    - verbose:
        Kalau True, akan print debug distribusi timestamp.

    Return:
        DataFrame:
            machine_id (str uppercase, trimmed)
            ts (datetime64[ns])
            value (float)
    """

    # 1. baca csv mentah
    df = pd.read_csv(csv_path)

    # 2. map kolom fleksibel ke ["machine_id","ts","value"]
    col_map_detected = _build_col_map(df)

    # rename sesuai hasil deteksi
    df = df.rename(columns=col_map_detected)

    required_cols = {"machine_id", "ts", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Kolom wajib tidak lengkap. Kolom ada: {list(df.columns)}; "
            f"harus ada minimal {required_cols}"
        )

    df = df[["machine_id", "ts", "value"]].copy()

    # 3. normalisasi machine_id → uppercase rapi
    df["machine_id"] = (
        df["machine_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # 4. value → float aman (support koma sebagai decimal separator)
    if df["value"].dtype == object:
        df["value"] = (
            df["value"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 5. parse timestamp
    # catatan: awalnya di kode kamu kita PAKSA dayfirst=False karena format sumber
    # ternyata MM/DD/YYYY HH:MM. Ini kita expose via argumen assume_us_datetime.
    df["ts"] = pd.to_datetime(
        df["ts"],
        errors="coerce",
        dayfirst=not assume_us_datetime,   # True → DD/MM/YYYY, False → MM/DD/YYYY
        infer_datetime_format=True,
    )

    # 6. bersihkan duplikat dan NA
    df = (
        df.dropna(subset=["machine_id", "ts", "value"])
          .sort_values(["machine_id", "ts"])
          .drop_duplicates(subset=["machine_id", "ts"], keep="last")
          .reset_index(drop=True)
    )

    # 7. optional debug
    debug_ts_coverage(df, verbose=verbose)

    return df


# ---------------------------------
# Quick self-test / manual run
# ---------------------------------
if __name__ == "__main__":
    # Contoh pemakaian manual:
    test_path = "/workspace/df_raw2.csv"  # ganti sesuai lokasi file kamu

    df_ts = load_timeseries_from_df_raw2(
        csv_path=test_path,
        assume_us_datetime=True,   # False kalau timestamp kamu formatnya DD/MM/YYYY
        verbose=True,
    )

    print("df_ts.head():")
    print(df_ts.head())

    print("\nunique machine_id:", df_ts["machine_id"].unique())
    print("total rows:", len(df_ts))

# =======================================================================================
# Cara PAKAI
# from other_ts_loader import load_timeseries_from_df_raw2
# from anomaly_query_engine import run_anomaly_query
# from intent_runtime import AnomalyQuadRunner
# from time_parse import resolve_time_window

# # 1. load data timeseries & thresholds
# df_ts = load_timeseries_from_df_raw2("/workspace/df_raw2.csv", assume_us_datetime=True)
# df_thr = load_thresholds("thresholds.xlsx")  # dari modul anomaly_range_pipeline kita sebelumnya

# # 2. siapin intent runner
# runner = AnomalyQuadRunner(
#     scope_dir="runs_anomaly_quad_relclass/scope/final_model",
#     gran_dir="runs_anomaly_quad_relclass/time_granularity/final_model",
#     cplx_dir="runs_anomaly_quad_relclass/time_complexity/final_model",
#     case_dir="runs_anomaly_quad_relclass/case/final_model",
# )

# # 3. jalankan query user end-to-end
# q = "Apakah ada anomali mesin XP888A 7 hari terakhir?"
# result = run_anomaly_query(
#     text=q,
#     runner=runner,
#     resolve_time_window_fn=resolve_time_window,
#     df_ts=df_ts,
#     df_thr=df_thr,
# )

# print(result.summary)
# print(result.details)
# =======================================================================================

# import pandas as pd
# from datetime import datetime

# # =========================
# # util bantu
# # =========================
# def ensure_datetime(series) -> pd.Series:
#     """
#     Konversi kolom waktu jadi datetime64[ns].
#     Mencoba format umum: ISO, dd/mm/yyyy HH:MM, dsb.
#     dayfirst=True supaya '01/02/2025' dibaca 1 Feb.
#     """
#     return pd.to_datetime(series, errors="coerce", dayfirst=True)


# # =========================
# # loader spesifik df_raw2.csv
# # =========================
# import pandas as pd

# def load_timeseries_from_df_raw2(csv_path: str) -> pd.DataFrame:
#     """
#     Baca df_raw2.csv jadi dataframe clean:
#     - machine_id (string uppercase)
#     - ts (datetime benar)
#     - value (float)
#     """
#     # 1. Baca csv
#     df = pd.read_csv(csv_path)

#     # 2. Map kolom (fallback kalau nama beda-beda)
#     def _norm_colname(name: str) -> str:
#         return (
#             str(name)
#             .strip()
#             .lower()
#             .replace(" ", "")
#             .replace("_", "")
#             .replace("-", "")
#             .replace("/", "")
#         )

#     col_map = {}
#     for c in df.columns:
#         norm_c = _norm_colname(c)

#         if norm_c in {"machineid","machine","mesin","unit","tag","tagnumber","tagnumbe"}:
#             col_map[c] = "machine_id"
#         elif norm_c in {
#             "ts","timestamp","datetime","timestamptime",
#             "timestampdatetime","datetimestamp"
#         }:
#             col_map[c] = "ts"
#         elif (norm_c in {"value","nilai","val","pv","reading","processvalue"}) or \
#              ("value" in norm_c):
#             col_map[c] = "value"
#         elif c in {"machine_id", "ts", "value"}:
#             col_map[c] = c

#     df = df.rename(columns=col_map)

#     required_cols = {"machine_id","ts","value"}
#     if not required_cols.issubset(df.columns):
#         raise ValueError(
#             f"Kolom wajib tidak lengkap. Kolom ada: {list(df.columns)}; "
#             f"harus ada: {required_cols}"
#         )

#     df = df[["machine_id","ts","value"]].copy()

#     # machine_id → uppercase rapi
#     df["machine_id"] = (
#         df["machine_id"]
#         .astype(str)
#         .str.upper()
#         .str.strip()
#     )

#     # value → float aman (support koma)
#     if df["value"].dtype == object:
#         df["value"] = (
#             df["value"]
#             .astype(str)
#             .str.replace(",", ".", regex=False)
#             .str.replace(" ", "", regex=False)
#         )
#     df["value"] = pd.to_numeric(df["value"], errors="coerce")

#     # === PARSING TIMESTAMP YANG BENAR ===
#     # Kita coba asumsikan formatnya MM/DD/YYYY HH:MM[:SS]
#     # Jangan dayfirst!
#     df["ts"] = pd.to_datetime(
#         df["ts"],
#         errors="coerce",
#         dayfirst=False,               # <-- ini perubahan penting
#         infer_datetime_format=True    # biar cepat & akurat
#     )

#     # 4. Cleanup dasar
#     df = (
#         df.dropna(subset=["machine_id","ts","value"])
#           .sort_values(["machine_id","ts"])
#           .drop_duplicates(subset=["machine_id","ts"], keep="last")
#           .reset_index(drop=True)
#     )

#     # 5. Debug coverage untuk ngecek bener ga sekarang
#     debug_ts_coverage(df)

#     return df


# def debug_ts_coverage(df_ts, max_print=50):
#     # print("=== DEBUG TS COVERAGE ===")
#     # print("dtypes :", df_ts.dtypes.to_dict())
#     # print("ts.min:", df_ts["ts"].min(), "ts.max:", df_ts["ts"].max())
#     # print()

#     # vc_date = (
#     #     df_ts["ts"]
#     #     .dt.date
#     #     .value_counts()
#     #     .sort_index()
#     # )
#     # print("Jumlah baris per tanggal:")
#     # print(vc_date.head(max_print).to_string())
#     # print("unique days:", vc_date.index.nunique())
#     # print()

#     # vc_month = (
#     #     df_ts["ts"]
#     #     .dt.to_period("M")
#     #     .value_counts()
#     #     .sort_index()
#     # )
#     # print("Jumlah baris per bulan:")
#     # print(vc_month.to_string())

#     # print("\nDistribusi nomor-hari (1..31) dalam df_ts:")
#     # vc_daynum = df_ts["ts"].dt.day.value_counts().sort_index()
#     # print(vc_daynum.to_string())
#     # print("=========================\n")



# # =========================
# # contoh pakai
# # =========================
# # path CSV kamu (replace kalau beda lokasi)
# import importlib
# import pandas as pd
# importlib.reload(pd) 
# df_ts = load_timeseries_from_df_raw2("/workspace/df_raw2.csv")

# print("df_ts.head():")
# print(df_ts.head())

# print("\nunique machine_id:", df_ts["machine_id"].unique())
# print("total rows:", len(df_ts))
