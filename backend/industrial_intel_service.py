# industrial_intel_service.py
#
# API minimal yang penting jalan.
# - bootstrap_all() bakal load semua resource sekali
# - /infer_case1  -> ringkasan/tren historis
# - /infer_case2  -> anomaly window / intent 4-head (case2)
# - /infer_case3  -> forecast & ETA turun threshold
#
# === YANG PERLU KAMU COCOKKAN ===
# - PATH file (excel_path, csv_save, TS_CSV_PATH di case2, dsb)
# - MODEL_ROOT / OUT_ROOT / FORECASTER_ROOT / DATA_CSV harus sama seperti di environment kamu.
#
# Setelah path sudah bener dan dependencies ke-import, langsung jalan.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd

# =========================
# CASE 1 IMPORT
# =========================
# dari kode kamu:
#   from data_loader import build_df_raw
#   from case1_path_pipeline import run_query
from data_loader import build_df_raw
from case1_path_pipeline import run_query as run_query_case1

# =========================
# CASE 2 IMPORT
# =========================
# ini langsung dari pipeline_full.py kamu
import os
import pandas as pd
from typing import Any, Dict
from Case2.queryFull import (
    AnomalyQuadRunner,
    build_runner,
    load_sensor_timeseries,
    run_query_once,
    DF_THRESHOLDS,
    MODEL_ROOT,
    TS_CSV_PATH,
)

# pipeline_full.pretty_print_payload dipakai di demo aja,
# jadi gak wajib diimport kalau ga dipakai di API.


# =========================
# CASE 3 IMPORT
# =========================
# ini dari main.py case3
from Case3.config import OUT_ROOT, FORECASTER_ROOT, DATA_CSV
from Case3.nlu_runner import ForecastMultiRunner
from Case3.forecaster import HorizonForecasterClassic
from Case3.timeseries_state import load_raw_timeseries
from Case3.orchestrator import run_full_query as run_full_query_case3


# =========================
# FASTAPI APP
# =========================
app = FastAPI()

# =========================
# GLOBAL STATE
# supaya semua heavy model/data cuma diload sekali
# =========================
class GlobalState:
    # case1
    df_raw_case1= None

    # case2
    runner_case2= None
    df_ts_case2=None
    df_thr_case2 = None

    # case3
    runner_case3 = None
    forecaster_case3 = None
    df_raw_case3 = None

state = GlobalState()
from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent

# =========================
# BOOTSTRAP SEKALI
# =========================
def bootstrap_all():
    print("[bootstrap] mulai load semua resource...")

    # ---------- CASE 1 BOOTSTRAP ----------
    # sesuai contoh code case1 kamu:
    excel_path = BASE_DIR/"Machine_Value_Processed.xlsx"  # SESUAIKAN
    sheet_name = "SW_3min"
    csv_save   = BASE_DIR/"df_raw2.csv"                    # SESUAIKAN

    df_raw_case1, _meta = build_df_raw(
        excel_path=excel_path,
        sheet_name=sheet_name,
        save_csv=csv_save,
        with_lag=False,
    )
    state.df_raw_case1 = df_raw_case1
    print(f"[bootstrap] CASE1 df_raw rows: {len(state.df_raw_case1)}")

    # ---------- CASE 2 BOOTSTRAP ----------
    # dari pipeline_full.py kamu
    # MODEL_ROOT   = "./runs_anomaly_quad_relclass"
    # TS_CSV_PATH  = "/workspace/df_raw2.csv"
    # DF_THRESHOLDS = DataFrame thresholds
    state.runner_case2 = build_runner(MODEL_ROOT)
    state.df_ts_case2 = load_sensor_timeseries(TS_CSV_PATH)
    state.df_thr_case2 = DF_THRESHOLDS.copy()
    print(f"[bootstrap] CASE2 df_ts rows: {len(state.df_ts_case2)}")
    print(f"[bootstrap] CASE2 df_thr rows: {len(state.df_thr_case2)}")

    # ---------- CASE 3 BOOTSTRAP ----------
    # dari main.py case3
    state.runner_case3 = ForecastMultiRunner(root_dir=OUT_ROOT)
    state.forecaster_case3 = HorizonForecasterClassic(run_root=FORECASTER_ROOT)
    state.df_raw_case3 = load_raw_timeseries(DATA_CSV)
    print(f"[bootstrap] CASE3 df_raw rows: {len(state.df_raw_case3)}")

    print("[bootstrap] SEMUA MODEL & DATA SIAP ✅")


# =========================
# Pydantic request/response
# =========================
class InferRequest(BaseModel):
    query: str

class InferResponse(BaseModel):
    raw: Dict[str, Any]
    llmContext: Dict[str, Any]


# =========================
# Helper bikin llmContext ringkas per case
# biar Next.js punya "context buat prompt", tapi gak kirim full raw data.
# =========================

def build_llm_context_case1(full_result: Dict[str, Any]) -> Dict[str, Any]:
    # result dari run_query_case1(...) punya key:
    # - human_report
    # - intent
    # - llm_metadata
    # - plot_paths
    return {
        "intent": full_result.get("intent"),
        "summary_brief": full_result.get("human_report"),
        "metrics": full_result.get("llm_metadata"),
    }

def build_llm_context_case2(payload_case2: Dict[str, Any]) -> Dict[str, Any]:
    # payload_case2 = hasil run_query_once()
    #
    # run_query_once() bikin payload siap LLM lewat anomaly_result_to_payload()
    # bentuknya biasanya:
    # {
    #   "window": {...},
    #   "anomalies": [...],
    #   "machines": [...],
    #   "explanation": "...",
    #   ...
    # }
    #
    # Kita ambil highlight supaya LLM bisa jawab operator.
    return {
        "summary": {
            "machines": payload_case2.get("machines"),
            "anomalies_count": len(payload_case2.get("anomalies", [])),
        },
        "window": payload_case2.get("window"),
        "explanation": payload_case2.get("explanation"),
    }

def build_llm_context_case3(full_result: Dict[str, Any]) -> Dict[str, Any]:
    # full_result = result dari run_full_query_case3()
    # Biasanya berisi niat user (intent), prediksi kapan crossing, dsb.
    intent = full_result.get("intent", {})
    anomalies = full_result.get("anomalies", [])
    affected_machines = (
        list({a.get("machine_id") for a in anomalies}) if anomalies else []
    )

    return {
        "intent": intent,
        "forecast_summary": full_result.get("forecast", {}),
        "eta": full_result.get("eta", {}),
        "operator_briefing": full_result.get("operator_briefing"),
        "anomaly_overview": {
            "hasAnomaly": len(anomalies) > 0,
            "affectedMachines": affected_machines,
        },
    }


# =========================
# HEALTHCHECK
# =========================
@app.get("/ping")
def ping():
    return {"status": "ok"}


# =========================
# ENDPOINT CASE1
# =========================
@app.post("/infer_case1", response_model=InferResponse)
def infer_case1(req: InferRequest):
    """
    Case1 = ringkasan tren/statistik historis per mesin / per periode waktu.
    Panggil run_query_case1(query, df_raw_case1).
    """
    result_case1 = run_query_case1(
        req.query,
        state.df_raw_case1,
        plots_dir="./plots"
    )

    return {
        "raw": result_case1,
        "llmContext": build_llm_context_case1(result_case1),
    }


# =========================
# ENDPOINT CASE2
# =========================
@app.post("/infer_case2", response_model=InferResponse)
def infer_case2(req: InferRequest):
    """
    Case2 = anomaly QA.
    Satu kalimat user -> intent 4-head -> time window -> scan anomaly.
    Panggil run_query_once(query, runner, df_ts, df_thr).
    """

    payload_case2 = run_query_once(
        query_text=req.query,
        runner=state.runner_case2,
        df_ts=state.df_ts_case2,
        df_thr=state.df_thr_case2,
    )

    return {
        "raw": payload_case2,
        "llmContext": build_llm_context_case2(payload_case2),
    }


# =========================
# ENDPOINT CASE3
# =========================
@app.post("/infer_case3", response_model=InferResponse)
def infer_case3(req: InferRequest):
    """
    Case3 = forecasting / ETA crossing threshold
    "kapan XP888A turun di bawah 70 derajat?"
    Panggil run_full_query_case3(user_text, runner, df_raw, forecaster)
    """

    result_case3 = run_full_query_case3(
        user_text=req.query,
        runner=state.runner_case3,
        df_raw=state.df_raw_case3,
        forecaster=state.forecaster_case3,
        n_lags=3,
    )

    return {
        "raw": result_case3,
        "llmContext": build_llm_context_case3(result_case3),
    }
