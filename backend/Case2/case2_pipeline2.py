# ============================================================
# demo_pipeline.py
# Smoke test end-to-end:
#   user query (bahasa natural)
#   → intent model (4-head)
#   → time window parser
#   → anomaly query engine
#   → payload siap kirim ke LLM / UI
#
# Catatan:
# - Script ini buat QA/manual check. Bukan library production.
# - Harus dijalankan setelah model hasil training sudah disimpan.
# ============================================================

from __future__ import annotations
from typing import Dict, Any
import pandas as pd

# ---- import modul internal kita ----
from intent_runtime import AnomalyQuadRunner          # runtime 4-head classifier
from time_parse import resolve_time_window            # NLP → TimeWindow(start,end,...)
from anomaly_query_engine import run_anomaly_query    # orchestrator text→result
from other_ts_loader import load_timeseries_from_df_raw2  # load df_raw2.csv yang sudah dibersihin
# NOTE:
# kamu punya 2 util ini di notebook:
#   - anomaly_result_to_payload(res, df_ts)
#   - pretty_print_payload(payload)
# aku akan stub fungsi minimalisnya di bawah supaya script ini self-contained.


# ============================================================
# (0) Helper formatter (stub sementara)
# ============================================================

def anomaly_result_to_payload(res, df_ts: pd.DataFrame) -> Dict[str, Any]:
    """
    Ubah AnomalyResult jadi payload dict yang gampang dikirim ke LLM / UI.
    Ini versi minimal. Silakan ganti sama formatter final kamu kalau sudah ada.
    """
    window_info = {
        "start": res.start.isoformat(),
        "end": res.end.isoformat(),
        "scope": res.scope,
        "machines": res.machines,
    }

    # ringkasan per mesin
    summary_rows = []
    for _, row in res.summary.iterrows():
        summary_rows.append({
            "machine_id":     row["machine_id"],
            "total_points":   int(row["total_points"]),
            "anomalies":      int(row["anomalies"]),
            "has_anomaly":    bool(row["has_anomaly"]),
            "first_ts":       row["first_ts"].isoformat() if pd.notna(row["first_ts"]) else None,
            "last_ts":        row["last_ts"].isoformat() if pd.notna(row["last_ts"])  else None,
        })

    # detail anomali baris-per-baris
    detail_rows = []
    for _, row in res.details.iterrows():
        detail_rows.append({
            "machine_id":  row["machine_id"],
            "ts":          row["ts"].isoformat() if pd.notna(row["ts"]) else None,
            "value":       float(row["value"]),
            "min_value":   float(row["min_value"]) if pd.notna(row["min_value"]) else None,
            "max_value":   float(row["max_value"]) if pd.notna(row["max_value"]) else None,
            "is_low":      bool(row["is_low"]),
            "is_high":     bool(row["is_high"]),
        })

    return {
        "window": window_info,
        "summary_per_machine": summary_rows,
        "anomaly_details": detail_rows,
        "raw_rowcount_total": int(len(df_ts)),
        "raw_rowcount_window": int(
            df_ts[
                (df_ts["ts"] >= res.start) & (df_ts["ts"] <= res.end) &
                (df_ts["machine_id"].isin(res.machines))
            ].shape[0]
        ),
        "query_text": res.query_text,
    }


def pretty_print_payload(payload: Dict[str, Any]) -> None:
    """
    Print ringkas biar gampang dicek manusia saat QA.
    """
    w = payload["window"]
    print(f"[WINDOW] {w['start']} → {w['end']} | scope={w['scope']} | machines={w['machines']}")
    print(f"[RAW COUNTS] total_rows={payload['raw_rowcount_total']} window_rows={payload['raw_rowcount_window']}")
    print("\n[SUMMARY PER MACHINE]")
    for m in payload["summary_per_machine"]:
        print(
            f"  - {m['machine_id']}: "
            f"{m['anomalies']} anomalies / {m['total_points']} pts "
            f"(first={m['first_ts']}, last={m['last_ts']})"
        )

    if payload["anomaly_details"]:
        print("\n[DETAIL ANOMALI]")
        for d in payload["anomaly_details"][:10]:  # limit print 10 baris biar gak kebanyakan
            flag = "LOW" if d["is_low"] else ("HIGH" if d["is_high"] else "")
            print(
                f"  {d['ts']} {d['machine_id']} "
                f"val={d['value']} [{d['min_value']}, {d['max_value']}] {flag}"
            )
    else:
        print("\n[DETAIL ANOMALI] (tidak ada anomaly rows)")


# ============================================================
# (1) Config path / model directory
# ============================================================
from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent

OUT_ROOT = BASE_DIR/"runs_anomaly_quad_relclass"  # sama seperti di trainer dan runtime

# Path CSV sensor yang udah dibersihin via data_loader.py / other_ts_loader.py
TS_CSV_PATH = BASE_DIR/"df_raw2.csv"  # ganti sesuai lokasi real kamu

# NOTE:
# df_thr (threshold min/max) sekarang masih dummy hardcoded.
# Nanti ini harus diganti pake loader threshold resmi (misal load_thresholds()).
df_thr = pd.DataFrame({
    "machine_id": ["XP888A", "PQ667"],
    "max_value":  [10,       140],
    "min_value":  [0,        110],
})


# ============================================================
# (2) Main demo flow
# ============================================================

def main_demo():
    # 2.1 Load model intent (4 head)
    runner = AnomalyQuadRunner(
        scope_dir=f"{OUT_ROOT}/scope/final_model",
        gran_dir=f"{OUT_ROOT}/time_granularity/final_model",
        cplx_dir=f"{OUT_ROOT}/time_complexity/final_model",
        case_dir=f"{OUT_ROOT}/case/final_model",
    )

    # 2.2 Load data timeseries sensor
    # NOTE: kita asumsikan format tanggal dari historian adalah MM/DD/YYYY HH:MM
    # kalau ternyata DD/MM/YYYY ubah assume_us_datetime=False
    df_ts = load_timeseries_from_df_raw2(
        TS_CSV_PATH,
        assume_us_datetime=True,
        verbose=False,
    )

    print("\n=== DEMO PREDIKSI ===")

    # 2.3 List contoh pertanyaan user natural language
    examples = [
        "Apakah ada anomali mesin XP888A dan PQ667 7 hari terakhir?",
        "Tolong tampilkan anomali semua mesin bulan ini",
        "Cek anomali mesin XP888A tanggal 1–20 Januari 2025",
        "Cek anomali mesin XP888A tanggal 1–20 Januari dan februari 2025",
        "Cek anomali mesin  bulan Januari dan februari 2025",
        "Cek anomali mesin  bulan Januari 2025",
        "Cek anomali mesin  bulan februari 2025",
        "Ada anomali semua mesin minggu lalu?",
        "Periksa anomali shift malam untuk XP888A",
        "apakah ada anomali kemarin malam?",
        "apakah ada anomali di senin minggu lalu pada pukul 8 pagi hingga 10 pagi?",
        "dalam rentang 30 menit terakhir apakah ada keanehan pada mesin?",
    ]

    # 2.4 Loop semua query → run anomaly pipeline → pretty print
    for q in examples:
        try:
            # jalankan full pipeline anomaly
            res = run_anomaly_query(
                text=q,
                runner=runner,
                resolve_time_window_fn=resolve_time_window,
                df_ts=df_ts,
                df_thr=df_thr,
            )

            # bungkus jadi payload terstruktur (misal buat kirim ke LLM/UI)
            payload = anomaly_result_to_payload(res, df_ts)

            # tampilkan hasil ringkas
            print("\n--- QUERY:", q, "---")
            pretty_print_payload(payload)

        except Exception as e:
            print("\n--- QUERY:", q, "---")
            print("Lewati (error):", e)

    print("\nModel tersimpan di:", OUT_ROOT)


# ============================================================
# (3) Entry point
# ============================================================

if __name__ == "__main__":
    main_demo()


# OUT_ROOT   = "./runs_anomaly_quad_relclass"

# print("\n=== DEMO PREDIKSI ===")
# df_thr = pd.DataFrame({
#     "machine_id": ["XP888A","PQ667"],
#     "max_value": [10, 140],
#     "min_value": [0, 110],
# })

# # df_ts nanti harus sudah bentuk final (machine_id, ts[datetime64], value[float])
# df_ts  = load_timeseries_from_df_raw2("/workspace/df_raw2.csv")
# examples = [
#     "Apakah ada anomali mesin XP888A dan PQ667 7 hari terakhir?",
#     "Tolong tampilkan anomali semua mesin bulan ini",
#     "Cek anomali mesin XP888A tanggal 1–20 Januari 2025",
#     "Cek anomali mesin XP888A tanggal 1–20 Januari dan februari 2025",
#     "Cek anomali mesin  bulan Januari dan februari 2025",
#     "Cek anomali mesin  bulan Januari 2025",
#     "Cek anomali mesin  bulan februari 2025",
#     "Ada anomali semua mesin minggu lalu?",
#     "Periksa anomali shift malam untuk XP888A",
#     "apakah ada anomali kemarin malam?",
#     "apakah ada anomali di senin minggu lalu pada pukul 8 pagi hingga 10 pagi?",
#     "dalam rentang 30 menit terakhir apakah ada keanehan pada mesin?"
# ]
# for q in examples:
#     try:
#         # jalankan full pipeline anomaly
#         res = run_anomaly_query(q, runner, df_ts=df_ts, df_thr=df_thr)

#         # bungkus jadi payload terstruktur
#         payload = anomaly_result_to_payload(res, df_ts)

#         # debug dump payload yg akan kita kirim ke LLM
#         print("\n--- QUERY:", q, "---")
#         pretty_print_payload(payload)

#     except Exception as e:
#         print("\n--- QUERY:", q, "---")
#         print("Lewati (error):", e)

# print("\nModel tersimpan di:", OUT_ROOT)
