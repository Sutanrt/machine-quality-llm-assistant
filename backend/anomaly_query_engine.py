# ============================================================
# anomaly_query_engine.py
# Dari teks user → intent model → time window → tabel anomali
# ============================================================

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np


# -------------------------------------------------
# 0) Utilities: ekstrak machine_id & parse datetime
# -------------------------------------------------

# Pola ID mesin (huruf kapital + angka), contoh: XP888A, PQ667, ABC123X
RE_MACHINE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b")


def extract_machines_from_text(text: str) -> List[str]:
    """
    Ambil kandidat machine_id dari teks user.
    Kita uppercase semua dan deduplicate sambil preserve urutan kemunculan.
    """
    return list(dict.fromkeys(RE_MACHINE.findall(text.upper())))


def ensure_datetime(series: pd.Series) -> pd.Series:
    """
    Coba parse berbagai format ts (ISO, dd/mm/yyyy HH:MM, Excel-like).
    """
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


# -------------------------------------------------
# 1) Loader helper (opsional)
# -------------------------------------------------

def load_thresholds_from_excel(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Load tabel batas min–max dari Excel.

    Output kolom wajib:
        machine_id (str, uppercase)
        min_value (float)
        max_value (float)
    """
    raw = pd.read_excel(path, sheet_name=sheet)

    # Deteksi otomatis nama kolom
    col_map = {}
    for c in raw.columns:
        lc = str(c).strip().lower()
        if "tag" in lc or "machine" in lc or "mesin" in lc:
            col_map[c] = "machine_id"
        elif "min" in lc:
            col_map[c] = "min_value"
        elif "max" in lc:
            col_map[c] = "max_value"

    df = raw.rename(columns=col_map)[["machine_id", "min_value", "max_value"]].copy()

    df["machine_id"] = df["machine_id"].astype(str).str.upper().str.strip()
    df["min_value"] = pd.to_numeric(df["min_value"], errors="coerce")
    df["max_value"] = pd.to_numeric(df["max_value"], errors="coerce")

    df = df.dropna(subset=["machine_id", "min_value", "max_value"]).reset_index(drop=True)
    return df


def load_timeseries_from_csv(path: str) -> pd.DataFrame:
    """
    Contoh loader timeseries.

    Harus mengembalikan kolom:
        machine_id (str, uppercase)
        ts         (datetime64[ns])
        value      (float)
    """
    df = pd.read_csv(path)

    # Normalisasi nama kolom
    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower().replace(" ", "")
        if lc in {"machine_id", "machineid", "tag", "tagnumber"}:
            col_map[c] = "machine_id"
        elif lc in {"ts", "timestamp", "datetime", "timestamp/datetime"}:
            col_map[c] = "ts"
        elif "value" in lc:
            col_map[c] = "value"

    df = df.rename(columns=col_map)[["machine_id", "ts", "value"]].copy()

    # Normalisasi isi kolom
    df["machine_id"] = df["machine_id"].astype(str).str.upper().str.strip()

    # ganti koma desimal → titik
    if df["value"].dtype == object:
        df["value"] = (
            df["value"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["ts"] = ensure_datetime(df["ts"])

    df = df.dropna(subset=["machine_id", "ts", "value"])
    return df


# -------------------------------------------------
# 2) Core result container
# -------------------------------------------------

@dataclass
class AnomalyResult:
    """
    Bentuk output final untuk satu query user.
    - summary: dataframe ringkas per mesin
    - details: dataframe baris-baris anomali
    """
    query_text: str
    start: datetime
    end: datetime
    scope: str                      # "ALL" / "SUBSET"
    machines: List[str]             # mesin yang dianalisis
    summary: pd.DataFrame           # ringkasan per mesin
    details: pd.DataFrame           # baris-baris yang anomali saja


# -------------------------------------------------
# 3) Engine utama: filter time window, join threshold, flag anomaly
# -------------------------------------------------

def detect_anomalies_for_window(
    query_text: str,
    pred: Dict[str, Any],
    time_window,                      # TimeWindow dari resolve_time_window(...)
    df_ts: pd.DataFrame,              # kolom: machine_id, ts, value
    df_thr: pd.DataFrame,             # kolom: machine_id, min_value, max_value
    machines_override: Optional[List[str]] = None,
) -> AnomalyResult:
    """
    1. Tentukan mesin mana yang dianalisis.
    2. Filter df_ts berdasarkan rentang waktu dan mesin.
    3. Join ke threshold (min_value / max_value).
    4. Tandai baris mana yang keluar dari rentang → is_anom.
    5. Build summary per mesin.
    """

    # --- scope dari model
    scope = (pred.get("scope") or {}).get("label", "ALL")

    # --- mesin target
    if machines_override is not None:
        machines = [m.upper() for m in machines_override]
    else:
        # coba ekstrak mesin dari teks
        mentioned = extract_machines_from_text(query_text)

        if scope == "ALL" or not mentioned:
            # user minta semua mesin, atau user gak nyebut mesin
            machines = sorted(df_thr["machine_id"].unique().tolist())
        else:
            # user nyebut mesin → tapi cuma pakai yg ada thresholdnya
            thr_set = set(df_thr["machine_id"])
            machines = [m for m in mentioned if m in thr_set]

    # fallback kalau list mesin kosong
    if not machines:
        machines = sorted(df_ts["machine_id"].unique().tolist())

    # --- filter waktu & mesin di timeseries
    mask_time = (df_ts["ts"] >= time_window.start) & (df_ts["ts"] <= time_window.end)
    mask_mach = df_ts["machine_id"].isin(machines)
    df_win = df_ts.loc[mask_time & mask_mach].copy()

    # --- join thresholds
    thr = df_thr.copy()
    thr["machine_id"] = thr["machine_id"].astype(str).str.upper()
    df_win = df_win.merge(thr, on="machine_id", how="left")

    # --- flag anomaly
    df_win["is_low"]  = df_win["value"] < df_win["min_value"]
    df_win["is_high"] = df_win["value"] > df_win["max_value"]
    df_win["is_anom"] = df_win["is_low"] | df_win["is_high"]

    # --- summary per machine
    # total_points, anomalies, first_ts, last_ts
    agg = (
        df_win
        .groupby("machine_id")
        .agg(
            total_points=("value", "size"),
            anomalies=("is_anom", "sum"),
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
        )
        .reset_index()
    )
    agg["has_anomaly"] = agg["anomalies"] > 0

    # sort mesin: taruh yg ada anomali dulu
    agg = agg.sort_values(["has_anomaly", "machine_id"], ascending=[False, True])

    # --- details: hanya baris anomali
    details = (
        df_win.loc[
            df_win["is_anom"],
            ["machine_id", "ts", "value", "min_value", "max_value", "is_low", "is_high"],
        ]
        .sort_values(["machine_id", "ts"])
        .reset_index(drop=True)
    )

    return AnomalyResult(
        query_text=query_text,
        start=time_window.start,
        end=time_window.end,
        scope=scope,
        machines=machines,
        summary=agg,
        details=details,
    )


# -------------------------------------------------
# 4) High-level wrapper: text → pred → time_window → anomalies
# -------------------------------------------------

from typing import Callable, Optional, Dict, Any
from datetime import datetime
import pandas as pd

# asumsi import yang lain tetap sama:
# - smart_predict
# - resolve_time_window
# - detect_anomalies_for_window
# - AnomalyResult dataclass

def run_anomaly_query(
    text: str,
    runner,
    df_ts: pd.DataFrame,
    df_thr: pd.DataFrame,

    # --- injection hooks baru: ---
    pred_override: Optional[Dict[str, Any]] = None,
    resolve_time_window_fn: Optional[
        Callable[[str, Dict[str, Any]], "TimeWindow"]
    ] = None,
) -> AnomalyResult:
    """
    Jalankan full pipeline anomalinya untuk satu query user.

    Args
    ----
    text : str
        Pertanyaan user dalam bahasa natural.
    runner : AnomalyQuadRunner
        Model 4-head yang sudah diload.
    df_ts : pd.DataFrame
        Timeseries sensor, kolom: machine_id, ts (datetime64), value (float)
    df_thr : pd.DataFrame
        Threshold min/max per machine_id.

    pred_override : dict (opsional)
        Jika sudah punya hasil smart_predict(text, runner) dari luar,
        kirim di sini supaya kita tidak hitung ulang.

    resolve_time_window_fn : callable (opsional)
        Fungsi custom yang menerima (text, pred_dict) dan
        harus return TimeWindow(start, end, granularity, kind, note).
        Kalau tidak diberikan, kita fallback ke resolve_time_window() bawaan lama.

    Returns
    -------
    AnomalyResult
        Objek struktur penuh (query_text, start, end, scope, machines, summary, details)
    """

    # 1) intent classification / routing
    if pred_override is not None:
        pred = pred_override
    else:
        pred = smart_predict(text, runner)

    # 2) time window resolution
    if resolve_time_window_fn is not None:
        tw = resolve_time_window_fn(text, pred)
    else:
        # default behaviour lama
        tw = resolve_time_window(text, pred)

    # 3) jalankan deteksi anomali di rentang tsb
    result = detect_anomalies_for_window(
        query_text=text,
        pred=pred,
        time_window=tw,
        df_ts=df_ts,
        df_thr=df_thr,
        machines_override=None,  # bisa diisi kalau mau paksa mesin tertentu
    )

    return result


# =====================
# Cara Pakai
# from intent_runtime import AnomalyQuadRunner
# from time_parse import resolve_time_window
# from anomaly_query_engine import run_anomaly_query

# result = run_anomaly_query(
#     text=user_query,
#     runner=quad_runner,
#     resolve_time_window_fn=resolve_time_window,
#     df_ts=df_ts,
#     df_thr=df_thr,
# )
# =======================

# # ============================================================
# # Anomaly Query Engine: dari teks → query → tabel anomali
# # ============================================================
# from __future__ import annotations
# import re
# from dataclasses import dataclass
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# import pandas as pd
# import numpy as np

# # ----------------------------
# # 0) UTIL — entity mesin & dt
# # ----------------------------
# RE_MACHINE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b")

# def extract_machines_from_text(text: str) -> List[str]:
#     """Ekstrak ID mesin seperti XP888A, PQ667, dsb."""
#     return list(dict.fromkeys(RE_MACHINE.findall(text.upper())))

# def ensure_datetime(series) -> pd.Series:
#     """Baca kolom waktu yang bisa berupa string ISO, dd/mm/yyyy HH:MM, atau Excel-like."""
#     # ganti koma → titik kalau ada (untuk angka yang tak sengaja masuk)
#     return pd.to_datetime(series, errors="coerce", dayfirst=True)

# # ---------------------------------------------------------
# # 1) LOADER (opsional) — ubah ke sumber data kamu sendiri
# # ---------------------------------------------------------
# def load_thresholds_from_excel(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
#     """
#     Membaca tabel batas min–max.
#     Harus menghasilkan kolom: machine_id, min_value, max_value
#     """
#     raw = pd.read_excel(path, sheet_name=sheet)
#     # DETEKSI otomatis nama kolom yang mirip
#     col_map = {}
#     for c in raw.columns:
#         lc = str(c).strip().lower()
#         if "tag" in lc or "machine" in lc: col_map[c] = "machine_id"
#         elif "min" in lc:                  col_map[c] = "min_value"
#         elif "max" in lc:                  col_map[c] = "max_value"
#     df = raw.rename(columns=col_map)[["machine_id","min_value","max_value"]].copy()
#     df["machine_id"] = df["machine_id"].astype(str).str.upper().str.strip()
#     return df

# def load_timeseries_from_csv(path: str) -> pd.DataFrame:
#     """
#     Contoh loader timeseries. Harus hasilkan kolom: machine_id, ts, value
#     """
#     df = pd.read_csv(path)
#     # normalisasi nama kolom
#     col_map = {}
#     for c in df.columns:
#         lc = str(c).strip().lower().replace(" ", "")
#         if lc in {"machine_id","machineid","tag","tagnumber"}: col_map[c] = "machine_id"
#         elif lc in {"ts","timestamp","datetime","timestamp/datetime"}: col_map[c] = "ts"
#         elif "value" in lc: col_map[c] = "value"
#     df = df.rename(columns=col_map)[["machine_id","ts","value"]].copy()

#     # normalisasi nilai
#     df["machine_id"] = df["machine_id"].astype(str).str.upper().str.strip()
#     # ganti koma desimal → titik bila perlu
#     if df["value"].dtype == object:
#         df["value"] = (df["value"].astype(str)
#                                  .str.replace(",", ".", regex=False)
#                                  .str.replace(" ", "", regex=False))
#     df["value"] = pd.to_numeric(df["value"], errors="coerce")
#     df["ts"] = ensure_datetime(df["ts"])
#     df = df.dropna(subset=["machine_id","ts","value"])
#     return df

# # -------------------------------------------------------------------
# # 2) CORE — filter data, join thresholds, dan tandai baris anomali
# # -------------------------------------------------------------------
# @dataclass
# class AnomalyResult:
#     query_text: str
#     start: datetime
#     end: datetime
#     scope: str
#     machines: List[str]
#     summary: pd.DataFrame
#     details: pd.DataFrame
    
# def debug_ts_coverage(df_ts):
#     print()
#     # print("Dtypes:", df_ts.dtypes.to_dict())
#     # print("ts.min:", df_ts["ts"].min(), "ts.max:", df_ts["ts"].max())
#     # vc = df_ts["ts"].dt.date.value_counts().sort_index()
#     # print("Jumlah baris per tanggal (top 40):")
#     # print(vc.head(40).to_string())

# def detect_anomalies_for_window(
#     query_text: str,
#     pred: Dict[str, Any],
#     tw,                              # TimeWindow dari resolve_time_window
#     df_ts: pd.DataFrame,             # kolom: machine_id, ts, value
#     df_thr: pd.DataFrame,            # kolom: machine_id, min_value, max_value
#     machines_override: Optional[List[str]] = None,
# ) -> AnomalyResult:
#     """
#     Jalankan deteksi anomali berdasarkan rentang waktu & mesin.
#     """
#     scope = (pred.get("scope") or {}).get("label", "ALL")

#     # Mesin dari teks; jika scope ALL, pakai semua dari thresholds
#     if machines_override is not None:
#         machines = [m.upper() for m in machines_override]
#     else:
#         found = extract_machines_from_text(query_text)
#         if scope == "ALL" or not found:
#             machines = sorted(df_thr["machine_id"].unique().tolist())
#         else:
#             # batasi hanya yang ada di thresholds
#             machines = [m for m in found if m in set(df_thr["machine_id"])]

#     if not machines:
#         # jika tidak ada sama sekali di thresholds, pakai semua yg ada di data
#         machines = sorted(df_ts["machine_id"].unique().tolist())
#     # debug_ts_coverage(df_ts)
#     # Filter waktu & mesin
#     mask_time = (df_ts["ts"] >= tw.start) & (df_ts["ts"] <= tw.end)
#     mask_mach = df_ts["machine_id"].isin(machines)
#     df_win = df_ts.loc[mask_time & mask_mach].copy()

#     # Join thresholds
#     thr = df_thr.copy()
#     thr["machine_id"] = thr["machine_id"].astype(str).str.upper()
#     df_win = df_win.merge(thr, on="machine_id", how="left")

#     # Tandai anomali
#     df_win["is_low"]  = df_win["value"] < df_win["min_value"]
#     df_win["is_high"] = df_win["value"] > df_win["max_value"]
#     df_win["is_anom"] = df_win["is_low"] | df_win["is_high"]
#     # 1) Apakah semua ts jatuh di hari ke-1?
#     days = df_ts["ts"].dt.day.value_counts().sort_index()
#     # print("Distribusi hari dalam sebulan:\n", days.to_string())
    
#     # 2) Apakah ada jejak operasi period/bulanan?
#     # (Cari string mencurigakan di file/nb code kamu)
#     suspicious = [
#         "to_period('M')", "to_period(\"M\")",
#         "astype('datetime64[M]')", 'astype("datetime64[M]")',
#         "floor('MS')", 'floor("MS")',
#         "floor('M')", 'floor("M")',
#         "Grouper(", "freq='M'", 'freq="M"', "freq='MS'", 'freq="MS"'
#     ]
#     # manual search di editor kamu; ini hanya daftar yang perlu dicari
#     print("Cari di kode: ", suspicious)

#     # Siapkan ringkasan
#     # - hitung total baris, total anomali, first/last ts per mesin
#     agg = (df_win
#            .groupby("machine_id")
#            .agg(total_points=("value","size"),
#                 anomalies=("is_anom","sum"),
#                 first_ts=("ts","min"),
#                 last_ts=("ts","max"))
#            .reset_index())
#     # Tambahkan flag “ada anomali?”
#     agg["has_anomaly"] = agg["anomalies"] > 0

#     # Detil anomali saja
#     details = (df_win.loc[df_win["is_anom"],
#                           ["machine_id","ts","value","min_value","max_value","is_low","is_high"]]
#                       .sort_values(["machine_id","ts"]))
    
#     return AnomalyResult(
#         query_text=query_text,
#         start=tw.start,
#         end=tw.end,
#         scope=scope,
#         machines=machines,
#         summary=agg.sort_values(["has_anomaly","machine_id"], ascending=[False,True]),
#         details=details
#     )

# # -------------------------------------------------------------------
# # 3) WRAPPER — dari teks langsung ke hasil
# # -------------------------------------------------------------------
# def run_anomaly_query(
#     text: str,
#     runner,                         # AnomalyQuadRunner milikmu
#     df_ts: pd.DataFrame,
#     df_thr: pd.DataFrame,
# ) -> AnomalyResult:
#     """
#     1) smart_predict → 2) resolve_time_window → 3) detect_anomalies_for_window
#     """
#     # print("range:", df_ts["ts"].min(), "→", df_ts["ts"].max())
#     # print(df_ts["ts"].dt.date.value_counts().sort_index().head(40))
#     # print("unique days:", df_ts["ts"].dt.date.nunique())
#     # print(df_ts["ts"].dt.day.value_counts().sort_index())  # kalau nyangkut di '1', ketahuan

#     pred = smart_predict(text, runner)
#     tw   = resolve_time_window(text, pred)
#     res  = detect_anomalies_for_window(text, pred, tw, df_ts, df_thr)
#     return res

# # -------------------------------------------------------------------
# # 4) CONTOH PEMAKAIAN
# # -------------------------------------------------------------------
# # (a) Kalau datamu sdh ada sebagai DataFrame, lompat ke (c)
# # df_thr = load_thresholds_from_excel("thresholds.xlsx", sheet="Sheet1")
# # df_ts  = load_timeseries_from_csv("timeseries.csv")

# # (b) Pastikan kolom ts bertipe datetime dan machine_id uppercase
# # df_ts["ts"] = ensure_datetime(df_ts["ts"])
# # df_ts["machine_id"] = df_ts["machine_id"].str.upper().str.strip()
# # df_thr["machine_id"] = df_thr["machine_id"].str.upper().str.strip()

# # (c) Jalankan query
# # q = "Apakah ada anomali mesin XP888A dan PQ667 7 hari terakhir?"
# # result = run_anomaly_query(q, runner, df_ts, df_thr)

# # (d) Akses hasil
# # print("WINDOW :", result.start, "→", result.end)
# # print("MESIN   :", result.machines)
# # print("\nRINGKASAN:")
# # print(result.summary)
# # print("\nDETIL ANOMALI:")
# # print(result.details)
