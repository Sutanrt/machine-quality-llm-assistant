# ============================================================
# data_loader.py
# - Baca semua sheet dari Excel sensor
# - Deteksi kolom machine_id / ts / value secara fleksibel
# - Normalisasi (uppercase machine, parse waktu, angka koma)
# - Diagnostics optional
# - Simpan CSV siap pakai untuk engine anomaly
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ---------------------------------
# Helpers: normalisasi nama kolom
# ---------------------------------

def _norm_col_name(s: str) -> str:
    """
    Normalisasi nama kolom biar bisa dicocokkan longgar.
    e.g. "Time Stamp / Date" -> "timestampdate"
    """
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace(":", "")
    )


# Kandidat nama kolom
CAND_MACHINE = {
    "tagnumber", "tagnumbe", "tag", "tagid",
    "machine", "mesin", "unit", "machineid"
}

# Gabungan kandidat timestamp yang sebelumnya banyak duplikat
CAND_TIMESTAMP = {
    "timestamp", "timestamps", "timestampdatetime", "timestampdatetim",
    "datetime", "datetimestamp", "waktu", "time", "date",
    "timestamptime", "timestamptimestamp",
    "timestampdatetime", "timestampdatetime",  # tetap allow repetisi biasa
}

CAND_VALUE = {
    "value", "nilai", "val", "reading", "pv", "processvalue"
}


def _guess_core_columns(df_sheet: pd.DataFrame) -> Dict[str, str]:
    """
    Cari kolom machine_id / ts / value di satu sheet.
    Return:
        {
            "machine_id": <original_col_name>,
            "ts": <original_col_name>,
            "value": <original_col_name>,
        }
    atau {} kalau gak ketemu lengkap.
    """
    normed = {c: _norm_col_name(c) for c in df_sheet.columns}

    # balik jadi {normalized_name: original_name_yang_pertama_ketemu}
    inv: Dict[str, str] = {}
    for orig, nrm in normed.items():
        inv.setdefault(nrm, orig)

    col_map: Dict[str, str] = {}

    # machine_id
    for nrm, orig in inv.items():
        if nrm in CAND_MACHINE:
            col_map["machine_id"] = orig
            break

    # timestamp
    for nrm, orig in inv.items():
        if nrm in CAND_TIMESTAMP:
            col_map["ts"] = orig
            break

    # value
    for nrm, orig in inv.items():
        if nrm in CAND_VALUE:
            col_map["value"] = orig
            break

    needed = {"machine_id", "ts", "value"}
    if not needed.issubset(col_map):
        return {}

    return col_map


def _to_float_any(x):
    """
    Ubah string '123,45' -> 123.45 jadi float.
    Kalau x udah numeric biarin aja.
    Return NaN kalau gagal parse.
    """
    if isinstance(x, str):
        x = x.replace(",", ".").replace(" ", "")
    return pd.to_numeric(x, errors="coerce")


def _clean_single_sheet(df_sheet: pd.DataFrame, sheet_name: str, *, verbose: bool = True) -> pd.DataFrame:
    """
    Bersihin satu sheet Excel:
    - deteksi kolom penting
    - rename -> machine_id, ts, value
    - parse ts
    - normalize angka
    - uppercase machine_id
    - sort dan dedup (per machine_id+ts terakhir menang)
    - tambahin kolom source_sheet

    Kalau sheet gak cocok, balikin df kosong (dengan kolom machine_id, ts, value).
    """
    cmap = _guess_core_columns(df_sheet)
    if not cmap:
        if verbose:
            print(
                f"[SKIP] Sheet '{sheet_name}' tidak punya kolom wajib "
                f"(machine_id / ts / value). Kolom yg ada: {list(df_sheet.columns)}"
            )
        return pd.DataFrame(columns=["machine_id", "ts", "value"])

    sub = df_sheet[[cmap["machine_id"], cmap["ts"], cmap["value"]]].copy()
    sub.columns = ["machine_id", "ts", "value"]

    # parse waktu (dayfirst=True biar dukung '12/01/2025 07:00')
    sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce", dayfirst=True)

    # normalisasi angka
    sub["value"] = sub["value"].apply(_to_float_any)

    # normalisasi machine_id
    sub["machine_id"] = sub["machine_id"].astype(str).str.upper().str.strip()

    # buang row kosong
    sub = sub.dropna(subset=["ts", "value", "machine_id"]).copy()

    # sort & dedup per mesin+timestamp
    sub = (
        sub.sort_values(["machine_id", "ts"])
        .drop_duplicates(subset=["machine_id", "ts"], keep="last")
        .reset_index(drop=True)
    )

    sub["source_sheet"] = sheet_name
    if verbose:
        print(f"  -> {sheet_name}: {len(sub)} rows setelah clean")

    return sub


# ---------------------------------
# Core ETL: Excel -> DF bersih
# ---------------------------------

def load_all_sheets_from_excel(
    excel_path: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Baca semua sheet dari `excel_path` dan gabungkan jadi 1 dataframe standar:
    kolom: machine_id (UPPER), ts (datetime64[ns]), value (float), source_sheet (str)

    Raises RuntimeError kalau gak ada satupun baris valid.
    """
    xls = pd.ExcelFile(excel_path)
    if verbose:
        print("Sheets terdeteksi:", xls.sheet_names)

    all_clean: List[pd.DataFrame] = []

    for sh in xls.sheet_names:
        raw_sheet = pd.read_excel(xls, sheet_name=sh)
        if verbose:
            print(f"\n=== Processing sheet: {sh} ===")
            print("Kolom asli:", list(raw_sheet.columns))

        cleaned = _clean_single_sheet(raw_sheet, sh, verbose=verbose)
        all_clean.append(cleaned)

    df_raw = pd.concat(all_clean, ignore_index=True)

    if df_raw.empty:
        raise RuntimeError(
            "Tidak ada data valid setelah proses semua sheet. "
            "Cek nama kolom di Excel."
        )

    # jaga konsistensi global sekali lagi
    df_raw["ts"] = pd.to_datetime(df_raw["ts"], errors="coerce")
    df_raw = df_raw.dropna(subset=["ts", "value", "machine_id"])

    df_raw = (
        df_raw.sort_values(["machine_id", "ts"])
        .drop_duplicates(subset=["machine_id", "ts"], keep="last")
        .reset_index(drop=True)
    )

    if verbose:
        print("\n=== DF_RAW FINAL (HEAD) ===")
        print(df_raw.head(10))
        print("shape:", df_raw.shape)

    return df_raw


# ---------------------------------
# Diagnostics helper (optional)
# ---------------------------------

def diagnose_timeseries(df_raw: pd.DataFrame, *, verbose: bool = True) -> dict:
    """
    Hitung statistik global untuk ngecek kualitas data sensor.
    Return dictionary ringkas. Optionally print detail.

    Output keys:
      - machines_unique
      - ts_min, ts_max
      - unique_days
      - per_day_sample_stats (describe() utk jumlah titik per hari per mesin)
      - freq_report (delta waktu paling umum per mesin)
    """
    out: dict = {}

    machines_unique = df_raw["machine_id"].unique().tolist()
    ts_min = df_raw["ts"].min()
    ts_max = df_raw["ts"].max()
    unique_days = df_raw["ts"].dt.date.nunique()

    # jumlah titik per (hari, mesin)
    per_day = (
        df_raw
        .groupby([df_raw["ts"].dt.date, "machine_id"])
        .size()
        .rename("count_per_day")
        .reset_index()
    )

    # statistik harian per mesin
    daily_stats = per_day.groupby("machine_id")["count_per_day"].describe()

    # coverage per tanggal (ignore mesin)
    by_date = df_raw.groupby(df_raw["ts"].dt.date).size()

    # cek interval antar-sample
    def _top_diffs(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("ts")
        diffs = g["ts"].diff().dropna()
        return diffs.value_counts().head(5)

    freq_report = {}
    for m_id, g in df_raw.groupby("machine_id"):
        freq_report[m_id] = _top_diffs(g)

    # rata-rata selang menit antar sampel per mesin
    freq_summary = {}
    for m_id, g in df_raw.groupby("machine_id"):
        g = g.sort_values("ts")
        diffs_min = g["ts"].diff().dropna().dt.total_seconds() / 60.0
        if len(diffs_min):
            freq_summary[m_id] = {
                "avg_step_min": float(diffs_min.mean()),
                "std_step_min": float(diffs_min.std()),
                "n": int(len(diffs_min)),
            }
        else:
            freq_summary[m_id] = {
                "avg_step_min": None,
                "std_step_min": None,
                "n": 0,
            }

    out["machines_unique"] = machines_unique
    out["ts_min"] = ts_min
    out["ts_max"] = ts_max
    out["unique_days"] = unique_days
    out["per_day_sample_stats"] = daily_stats
    out["by_date_counts"] = by_date
    out["freq_report_topdiffs"] = freq_report
    out["freq_summary_minutes"] = freq_summary

    if verbose:
        print("\n=== DIAGNOSTIK GLOBAL ===")
        print("Mesin unik:", machines_unique)
        print("Rentang waktu global :", ts_min, "→", ts_max)
        print("Jumlah hari unik     :", unique_days)

        print("\nContoh distribusi jumlah titik per (hari, mesin):")
        print(per_day.head(20))

        print("\nStatistik jumlah titik /hari /mesin:")
        print(daily_stats)

        print("\nJumlah row per hari (first 60 hari):")
        print(by_date.head(60))

        print("\n=== CEK INTERVAL SAMPLING PER MESIN ===")
        for m_id, vc in freq_report.items():
            print(f"\nMesin {m_id} diff(ts) paling sering muncul:")
            print(vc)

        for m_id, stats in freq_summary.items():
            if stats["n"] > 0:
                print(
                    f"\nMesin {m_id} avg step (menit): "
                    f"{stats['avg_step_min']:.2f} "
                    f"(std {stats['std_step_min']:.2f}) n={stats['n']}"
                )
            else:
                print(f"\nMesin {m_id} hanya punya 1 titik waktu (nggak bisa hitung interval).")

    return out


# ---------------------------------
# Save helper
# ---------------------------------

def save_clean_csv(df_raw: pd.DataFrame, out_csv_path: str, *, verbose: bool = True) -> None:
    """
    Simpan dataframe hasil bersih ke CSV final supaya bisa dipakai engine.
    """
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(out_csv_path, index=False)
    if verbose:
        print(f"[DONE] Data bersih disimpan ke: {out_csv_path}")


# ---------------------------------
# Convenience high-level function
# ---------------------------------

def build_timeseries_from_excel(
    excel_path: str,
    out_csv_path: Optional[str] = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fungsi sekali jalan:
    1. load semua sheet
    2. run diagnostics
    3. optional: simpan CSV
    4. return df_raw (machine_id, ts, value, source_sheet)

    Ini basically pengganti script top-level di case2_dfraw_ts_extractor.py. :contentReference[oaicite:1]{index=1}
    """
    df_raw = load_all_sheets_from_excel(excel_path, verbose=verbose)
    diagnose_timeseries(df_raw, verbose=verbose)

    if out_csv_path:
        save_clean_csv(df_raw, out_csv_path, verbose=verbose)

    return df_raw
# =========================
# Cara Pakai
# from data_loader import build_timeseries_from_excel

# df_raw = build_timeseries_from_excel(
#     excel_path="/workspace/wefesfes2.xlsx",
#     out_csv_path="/workspace/df_raw2.csv",
#     verbose=True,
# )

# # df_raw punya kolom:
# #   machine_id | ts | value | source_sheet
# # tinggal pakai:
# df_ts = df_raw[["machine_id", "ts", "value"]].copy()
# ===========================
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # ======================================
# # CONFIG
# # ======================================
# EXCEL_PATH = "/workspace/wefesfes2.xlsx"   # update kalau filenya di path lain
# OUT_CSV    = "/workspace/df_raw2.csv"      # hasil bersih

# # ======================================
# # HELPER: normalisasi nama kolom
# # ======================================
# def norm(s: str) -> str:
#     return (
#         str(s)
#         .strip()
#         .lower()
#         .replace(" ", "")
#         .replace("_", "")
#         .replace("-", "")
#         .replace("/", "")
#         .replace(":", "")
#     )

# # kandidat nama kolom untuk mapping fleksibel
# CAND_MACHINE   = {"tagnumber", "tagnumbe", "tag", "machine", "mesin", "unit", "machineid", "tagid"}
# CAND_TIMESTAMP = {
#     "timestampdatetime", "timestampdatetime", "timestamp", "datetime", "waktu",
#     "timestamptime", "timestamptimestamp", "timestampdatetime", "timestampdatetime",
#     "timestampdatetime", "timestamptime", "datetime", "datetimestamp", "timestampdatetime"
# }
# # kita broad-kan sedikit biar aman
# CAND_TIMESTAMP = CAND_TIMESTAMP.union({"timestampdatetime","timestampdatetime","timestampdatetime","timestampdatetime","timestampdatetime","timestampdatetime","timestamps","time","date","datetimestamp","timestampdatetime","timestampdatetim"}) 
# CAND_VALUE     = {"value", "nilai", "val", "reading", "pv", "processvalue"}

# def map_columns(df_sheet: pd.DataFrame):
#     """
#     Cari kolom machine_id / ts / value di satu sheet.
#     Return dict {'machine_id': colA, 'ts': colB, 'value': colC}
#     atau {} kalau ga ketemu.
#     """
#     cols_norm = {c: norm(c) for c in df_sheet.columns}
#     # kebalikannya buat lookup cepat
#     inv = {}
#     for orig, nrm in cols_norm.items():
#         inv.setdefault(nrm, orig)

#     col_map = {}

#     # machine_id
#     for nrm, orig in inv.items():
#         if nrm in CAND_MACHINE:
#             col_map["machine_id"] = orig
#             break

#     # timestamp
#     for nrm, orig in inv.items():
#         if nrm in CAND_TIMESTAMP:
#             col_map["ts"] = orig
#             break

#     # value
#     for nrm, orig in inv.items():
#         if nrm in CAND_VALUE:
#             col_map["value"] = orig
#             break

#     needed = {"machine_id", "ts", "value"}
#     if not needed.issubset(col_map.keys()):
#         return {}

#     return col_map


# def to_float_any(x):
#     """
#     Ubah string '123,45' -> 123.45 (float)
#     Kalau sudah float/int biarin saja.
#     """
#     if isinstance(x, str):
#         x = x.replace(",", ".")
#     return pd.to_numeric(x, errors="coerce")


# def clean_one_sheet(df_sheet: pd.DataFrame, sheet_name: str):
#     """
#     - deteksi kolom penting
#     - rename -> ['machine_id','ts','value']
#     - parse waktu, normalisasi angka
#     - buang null
#     - return df bersih (index reset)
#     """
#     cmap = map_columns(df_sheet)
#     if not cmap:
#         print(f"[SKIP] Sheet '{sheet_name}' tidak punya kolom wajib (machine_id / ts / value). Kolom yg ada: {list(df_sheet.columns)}")
#         return pd.DataFrame(columns=["machine_id","ts","value"])

#     sub = df_sheet[[cmap["machine_id"], cmap["ts"], cmap["value"]]].copy()
#     sub.columns = ["machine_id", "ts", "value"]

#     # parse waktu
#     sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce", dayfirst=True)

#     # normalisasi nilai angka
#     sub["value"] = sub["value"].apply(to_float_any)

#     # bersihkan
#     sub["machine_id"] = sub["machine_id"].astype(str).str.upper().str.strip()
#     sub = sub.dropna(subset=["ts","value","machine_id"]).copy()

#     # sort & dedup per mesin+timestamp
#     sub = (sub
#            .sort_values(["machine_id","ts"])
#            .drop_duplicates(subset=["machine_id","ts"], keep="last")
#            .reset_index(drop=True))

#     sub["source_sheet"] = sheet_name  # supaya kita tahu asalnya
#     return sub


# # ======================================
# # 1. LOAD SEMUA SHEET
# # ======================================
# xls = pd.ExcelFile(EXCEL_PATH)
# all_clean = []

# print("Sheets terdeteksi:", xls.sheet_names)
# for sh in xls.sheet_names:
#     raw_sheet = pd.read_excel(xls, sheet_name=sh)
#     print(f"\n=== Processing sheet: {sh} ===")
#     print("Kolom asli:", list(raw_sheet.columns))

#     cleaned = clean_one_sheet(raw_sheet, sh)

#     print(f" -> Rows setelah clean: {len(cleaned)}")
#     if len(cleaned):
#         print(cleaned.head(3))
#     all_clean.append(cleaned)

# # gabung semua sheet
# df_raw = pd.concat(all_clean, ignore_index=True)

# # fallback kalau kosong total
# if df_raw.empty:
#     raise RuntimeError("Tidak ada data valid setelah proses semua sheet. Cek nama kolom di Excel.")

# # pastikan tipe
# df_raw["ts"] = pd.to_datetime(df_raw["ts"], errors="coerce")
# df_raw = df_raw.dropna(subset=["ts","value","machine_id"])

# # resort global dan dedup global
# df_raw = (
#     df_raw
#     .sort_values(["machine_id","ts"])
#     .drop_duplicates(subset=["machine_id","ts"], keep="last")
#     .reset_index(drop=True)
# )

# print("\n=== DF_RAW FINAL (HEAD) ===")
# print(df_raw.head(10))
# print("shape:", df_raw.shape)

# # ======================================
# # 2. DIAGNOSTIK GLOBAL
# # ======================================
# print("\n=== DIAGNOSTIK GLOBAL ===")
# machines_unique = df_raw["machine_id"].unique().tolist()
# print("Mesin unik:", machines_unique)

# ts_min = df_raw["ts"].min()
# ts_max = df_raw["ts"].max()
# print("Rentang waktu global : ", ts_min, "→", ts_max)

# # berapa hari unik?
# unique_days = df_raw["ts"].dt.date.nunique()
# print("Jumlah hari unik      :", unique_days)

# # group per tanggal & mesin → hitung point/hari per mesin
# per_day = (
#     df_raw
#     .groupby([df_raw["ts"].dt.date, "machine_id"])
#     .size()
#     .rename("count_per_day")
#     .reset_index()
# )

# print("\nContoh distribusi jumlah titik per (hari, mesin):")
# print(per_day.head(20))

# # estimasi rata-rata sample per hari per mesin
# daily_stats = per_day.groupby("machine_id")["count_per_day"].describe()
# print("\nStatistik jumlah titik /hari /mesin:")
# print(daily_stats)

# # coverage harian (berapa row per hari calendar, ignore mesin)
# by_date = df_raw.groupby(df_raw["ts"].dt.date).size()
# print("\nJumlah row per hari (first 60 hari):")
# print(by_date.head(60))

# # ======================================
# # 3. CEK INTERVAL (APAKAH ~5 MENIT?)
# # ======================================
# print("\n=== CEK INTERVAL SAMPLING PER MESIN ===")
# def check_freq(g):
#     g = g.sort_values("ts")
#     diffs = g["ts"].diff().dropna()
#     # ambil delta yg paling umum (mode approx)
#     top = (
#         diffs.value_counts()
#         .head(5)
#     )
#     return top

# freq_report = {}
# for m_id, g in df_raw.groupby("machine_id"):
#     freq_report[m_id] = check_freq(g)

# for m_id, vc in freq_report.items():
#     print(f"\nMesin {m_id} diff(ts) paling sering muncul:")
#     print(vc)

# # kalau ingin lihat apakah rata-rata diff ~5 menit:
# for m_id, g in df_raw.groupby("machine_id"):
#     g = g.sort_values("ts")
#     diffs_min = g["ts"].diff().dropna().dt.total_seconds() / 60.0
#     if len(diffs_min):
#         print(f"\nMesin {m_id} avg step (menit): {diffs_min.mean():.2f} (std {diffs_min.std():.2f}) n={len(diffs_min)})")
#     else:
#         print(f"\nMesin {m_id} hanya punya 1 titik waktu (nggak bisa hitung interval).")

# # ======================================
# # 4. SIMPAN KE CSV / DIPAKAI OLEH ENGINE
# # ======================================
# Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
# df_raw.to_csv(OUT_CSV, index=False)

# print(f"\n=== DONE ===")
# print(f"Hasil data bersih disimpan di: {OUT_CSV}")

# # kalau mau langsung pakai ke engine anomaly:
# # df_ts  = df_raw[["machine_id","ts","value"]].copy()
# # df_thr = <data threshold kamu>  # misal DataFrame manual min/max tiap mesin
