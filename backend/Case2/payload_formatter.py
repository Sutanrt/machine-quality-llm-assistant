# ============================================================
# payload_formatter.py
#
# Konversi hasil deteksi anomali (AnomalyResult) jadi payload
# JSON-friendly untuk dikirim ke LLM / frontend.
#
# Dipakai di demo_pipeline.py:
#   payload = anomaly_result_to_payload(res, df_ts)
#   pretty_print_payload(payload)
# ============================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import pandas as pd


# -------------------------------------------------
# helper kecil untuk aman format datetime ke string
# -------------------------------------------------

def _ts_to_str(x: Any) -> Optional[str]:
    """
    Convert datetime-like ke string "YYYY-MM-DD HH:MM:SS".
    Return None kalau kosong / NaT / NaN.
    """
    if pd.isna(x):
        return None
    # banyak objek datetime-like di pandas udah punya .strftime
    try:
        return x.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        return str(x)


# -------------------------------------------------
# Ringkasan per mesin
# -------------------------------------------------

def df_summary_to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert result.summary (DataFrame per-mesin) → list of dict.

    Field yang diambil:
      - machine_id
      - total_points
      - anomalies
      - first_ts
      - last_ts
      - has_anomaly

    Semua timestamp di-cast ke string ISO-like agar aman untuk json.dumps.
    """
    if df is None or len(df) == 0:
        return []

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        total_points = row.get("total_points", 0)
        anomalies = row.get("anomalies", 0)
        has_anomaly = row.get("has_anomaly", False)

        out.append({
            "machine_id":    str(row.get("machine_id")),
            "total_points":  int(total_points) if not pd.isna(total_points) else 0,
            "anomalies":     int(anomalies) if not pd.isna(anomalies) else 0,
            "first_ts":      _ts_to_str(row.get("first_ts")),
            "last_ts":       _ts_to_str(row.get("last_ts")),
            "has_anomaly":   bool(has_anomaly) if not pd.isna(has_anomaly) else False,
        })
    return out


# -------------------------------------------------
# Detail baris anomali
# -------------------------------------------------

def df_details_to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert result.details (baris anomali saja) → list of dict.

    Field yang diambil:
      - machine_id
      - ts
      - value
      - min_value
      - max_value
      - is_low
      - is_high
    """
    if df is None or len(df) == 0:
        return []

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append({
            "machine_id": str(row.get("machine_id")),
            "ts":         _ts_to_str(row.get("ts")),
            "value":      float(row.get("value")) if pd.notna(row.get("value")) else None,
            "min_value":  float(row.get("min_value")) if pd.notna(row.get("min_value")) else None,
            "max_value":  float(row.get("max_value")) if pd.notna(row.get("max_value")) else None,
            "is_low":     bool(row.get("is_low"))  if not pd.isna(row.get("is_low"))  else False,
            "is_high":    bool(row.get("is_high")) if not pd.isna(row.get("is_high")) else False,
        })
    return out


# -------------------------------------------------
# Info coverage dataset mentah
# -------------------------------------------------

def build_data_window_stats(
    df_ts: pd.DataFrame,
    sample_dates: int = 5,
) -> Dict[str, Any]:
    """
    Ambil konteks global dataset sensor (buat bantu LLM/frontend ngerti cakupan data).
    Menghasilkan:
      - ts_min / ts_max (string)
      - rows_total
      - rows_per_date_sample: { '2025-01-10': 432, ... }
        (bukan semua tanggal supaya payload gak bengkak)

    Cara sampling tanggal:
      - 2 tanggal terawal
      - 2 tanggal terakhir
      - 1 tanggal tengah (kalau ada)
    """
    if df_ts.empty:
        return {
            "ts_min": None,
            "ts_max": None,
            "rows_total": 0,
            "rows_per_date_sample": {},
        }

    ts_min = df_ts["ts"].min()
    ts_max = df_ts["ts"].max()
    rows_total = int(len(df_ts))

    # jumlah row per tanggal
    per_date_counts = (
        df_ts
        .groupby(df_ts["ts"].dt.date)
        .size()
        .rename("count")
        .sort_index()
    )

    unique_dates = per_date_counts.index.tolist()
    sample_idx: List[int] = []

    if len(unique_dates) > 0:
        sample_idx.append(0)  # tanggal paling awal
    if len(unique_dates) > 1:
        sample_idx.append(1)  # tanggal kedua paling awal
    if len(unique_dates) > 2:
        sample_idx.append(len(unique_dates) // 2)  # tanggal tengah
    if len(unique_dates) > 3:
        sample_idx.append(-2)  # tanggal kedua terakhir
    if len(unique_dates) > 4:
        sample_idx.append(-1)  # tanggal terakhir

    # deduplicate index terpilih sambil preserve order
    sample_idx = [unique_dates[i] for i in dict.fromkeys(sample_idx)]

    rows_per_date_sample: Dict[str, int] = {}
    for d in sample_idx[:sample_dates]:
        rows_per_date_sample[str(d)] = int(per_date_counts.loc[d])

    return {
        "ts_min": _ts_to_str(ts_min),
        "ts_max": _ts_to_str(ts_max),
        "rows_total": rows_total,
        "rows_per_date_sample": rows_per_date_sample,
    }


# -------------------------------------------------
# High-level formatter utama
# -------------------------------------------------

# NOTE:
# AnomalyResult didefinisikan di anomaly_query_engine.py
# dataclassnya:
#   query_text: str
#   start: datetime
#   end: datetime
#   scope: str
#   machines: List[str]
#   summary: pd.DataFrame
#   details: pd.DataFrame
#
# Kita gak re-declare dataclassnya di sini supaya gak circular import.
# Cukup pakai type comment di docstring.


def anomaly_result_to_payload(
    result,          # AnomalyResult
    df_ts: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Gabungkan jadi satu payload dict JSON-friendly:
      - info query (text user, mesin yg dianalisis, scope)
      - rentang waktu yang dipakai (start..end)
      - ringkasan per mesin (summary_per_machine)
      - detail anomaly (anomaly_events)
      - statistik coverage dataset global (data_window_stats)

    Output ini aman langsung untuk json.dumps(payload).

    result : AnomalyResult
    df_ts  : dataframe mentah sensor full (utk statistik konteks)
    """
    payload: Dict[str, Any] = {
        "query_text": result.query_text,
        "resolved_time_window": {
            "start": _ts_to_str(result.start),
            "end":   _ts_to_str(result.end),
        },
        "machines_considered": list(result.machines),
        "scope_label": result.scope,
        "data_window_stats": build_data_window_stats(df_ts),
        "summary_per_machine": df_summary_to_list(result.summary),
        "anomaly_events": df_details_to_list(result.details),
    }
    return payload


# -------------------------------------------------
# Pretty printer (debug / QA manual)
# -------------------------------------------------

def pretty_print_payload(payload: Dict[str, Any]) -> None:
    """
    Utility buat QA:
    tampilkan payload dengan indent 2 supaya manusia gampang baca.
    """
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# import json
# from typing import Any, Dict

# def df_summary_to_list(df: pd.DataFrame) -> list[Dict[str, Any]]:
#     """
#     Convert df_result.summary → list of dict
#     Kolom yang diambil: machine_id, total_points, anomalies, first_ts, last_ts, has_anomaly
#     Semua timestamp di-cast ke string ISO biar JSON friendly.
#     """
#     if df is None or len(df) == 0:
#         return []

#     out = []
#     for _, row in df.iterrows():
#         out.append({
#             "machine_id":    str(row.get("machine_id")),
#             "total_points":  int(row.get("total_points", 0)) if not pd.isna(row.get("total_points")) else 0,
#             "anomalies":     int(row.get("anomalies", 0)) if not pd.isna(row.get("anomalies")) else 0,
#             "first_ts":      row.get("first_ts").strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("first_ts")) else None,
#             "last_ts":       row.get("last_ts").strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("last_ts")) else None,
#             "has_anomaly":   bool(row.get("has_anomaly")) if not pd.isna(row.get("has_anomaly")) else False,
#         })
#     return out


# def df_details_to_list(df: pd.DataFrame) -> list[Dict[str, Any]]:
#     """
#     Convert df_result.details → list of dict anomaly events
#     Kolom yang diambil:
#       machine_id, ts, value, min_value, max_value, is_low, is_high
#     """
#     if df is None or len(df) == 0:
#         return []

#     out = []
#     for _, row in df.iterrows():
#         out.append({
#             "machine_id": str(row.get("machine_id")),
#             "ts": row.get("ts").strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("ts")) else None,
#             "value": float(row.get("value")) if pd.notna(row.get("value")) else None,
#             "min_value": float(row.get("min_value")) if pd.notna(row.get("min_value")) else None,
#             "max_value": float(row.get("max_value")) if pd.notna(row.get("max_value")) else None,
#             "is_low": bool(row.get("is_low")) if not pd.isna(row.get("is_low")) else False,
#             "is_high": bool(row.get("is_high")) if not pd.isna(row.get("is_high")) else False,
#         })
#     return out


# def build_data_window_stats(df_ts: pd.DataFrame, sample_dates: int = 5) -> Dict[str, Any]:
#     """
#     Ambil konteks global data (buat bantu LLM ngerti coverage data kita).
#     - ts_min / ts_max
#     - total rows
#     - beberapa contoh jumlah row / tanggal (bukan semua tanggal supaya tidak bengkak)
#     """
#     if df_ts.empty:
#         return {
#             "ts_min": None,
#             "ts_max": None,
#             "rows_total": 0,
#             "rows_per_date_sample": {}
#         }

#     ts_min = df_ts["ts"].min()
#     ts_max = df_ts["ts"].max()
#     rows_total = len(df_ts)

#     # hitung jumlah row per tanggal
#     per_date_counts = (
#         df_ts.groupby(df_ts["ts"].dt.date)
#              .size()
#              .rename("count")
#              .sort_index()
#     )

#     # ambil sample beberapa tanggal “representatif”:
#     # - 2 tanggal terawal
#     # - 2 tanggal terakhir
#     # - 1 tanggal tengah (kalau ada)
#     unique_dates = per_date_counts.index.tolist()
#     sample_idx = []
#     if len(unique_dates) > 0:
#         sample_idx.append(0)  # awal
#     if len(unique_dates) > 1:
#         sample_idx.append(1)  # kedua awal
#     if len(unique_dates) > 2:
#         sample_idx.append(len(unique_dates)//2)  # tengah
#     if len(unique_dates) > 3:
#         sample_idx.append(-2)  # kedua terakhir
#     if len(unique_dates) > 4:
#         sample_idx.append(-1)  # terakhir

#     # dedup index terpilih
#     sample_idx = [unique_dates[i] for i in dict.fromkeys(sample_idx)]
#     rows_per_date_sample = {}
#     for d in sample_idx[:sample_dates]:
#         rows_per_date_sample[str(d)] = int(per_date_counts.loc[d])

#     return {
#         "ts_min": ts_min.strftime("%Y-%m-%d %H:%M:%S"),
#         "ts_max": ts_max.strftime("%Y-%m-%d %H:%M:%S"),
#         "rows_total": int(rows_total),
#         "rows_per_date_sample": rows_per_date_sample,
#     }


# def anomaly_result_to_payload(
#     result: AnomalyResult,
#     df_ts: pd.DataFrame,
# ) -> Dict[str, Any]:
#     """
#     Gabungkan:
#     - info query (text, window, mesin)
#     - ringkasan per mesin (summary)
#     - daftar event anomaly (details)
#     - statistik coverage dataset df_ts

#     Return dict pure-Python yang aman untuk json.dumps.
#     """
#     payload = {
#         "query_text": result.query_text,
#         "resolved_time_window": {
#             "start": result.start.strftime("%Y-%m-%d %H:%M:%S"),
#             "end":   result.end.strftime("%Y-%m-%d %H:%M:%S"),
#         },
#         "machines_considered": list(result.machines),
#         "scope_label": result.scope,  # "ALL" / "NOT_ALL"
#         "data_window_stats": build_data_window_stats(df_ts),
#         "summary_per_machine": df_summary_to_list(result.summary),
#         "anomaly_events": df_details_to_list(result.details),
#     }
#     return payload


# # OPTIONAL HELPER:
# def pretty_print_payload(payload: Dict[str, Any]):
#     """
#     Hanya buat ngecek outputnya human readable.
#     """
#     print(json.dumps(payload, indent=2))
