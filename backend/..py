# industrial_intel_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Tuple
import pandas as pd

# ==== IMPORT CASE3 PIPELINE ====
from case3_pipeline.config import OUT_ROOT, FORECASTER_ROOT, DATA_CSV
from case3_pipeline.nlu_runner import ForecastMultiRunner
from case3_pipeline.forecaster import HorizonForecasterClassic
from case3_pipeline.timeseries_state import load_raw_timeseries
from case3_pipeline.orchestrator import run_full_query as run_full_query_case3

# ==== IMPORT CASE1 PIPELINE ====
# asumsi case1 sudah punya:
#   - build_df_raw()
#   - run_query(query_text, df_raw, plots_dir=...)
from data_loader import build_df_raw
from case1_path_pipeline import run_query as run_query_case1

# ==== IMPORT CASE2 PIPELINE ====
# placeholder: kamu isi nanti (misal analisis produksi, RnD, dsb.)
# Untuk ssekarang kita buat dummy supaya endpointnya udah ada.
def run_full_query_case2(user_text: str) -> Dict[str, Any]:
    """
    TODO: Ganti dengan pipeline case2 beneran.
    Sementara kita balikin dummy.
    """
    return {
        "intent": {"kind": "CASE2_PLACEHOLDER"},
        "note": "case2 belum diimplementasikan",
        "summary": "no data",
        "details": [],
    }

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI()

# -------------------------------------------------
# Global state buat bootstrap berat
# -------------------------------------------------

class GlobalState:
    # case1
    df_raw_case1: pd.DataFrame | None = None

    # case3
    runner_case3: ForecastMultiRunner | None = None
    forecaster_case3: HorizonForecasterClassic | None = None
    df_raw_case3: pd.DataFrame | None = None

state = GlobalState()

@app.on_event("startup")
def bootstrap_all():
    # -------- CASE1 BOOTSTRAP --------
    # load df_raw sekali
    excel_path = "Machine_Value_Processed.xlsx"
    sheet_name = "SW_3min"
    csv_save   = "df_raw.csv"

    df_raw_case1, _meta = build_df_raw(
        excel_path=excel_path,
        sheet_name=sheet_name,
        save_csv=csv_save,
        with_lag=False,
    )
    state.df_raw_case1 = df_raw_case1
    print(f"[bootstrap] CASE1 df_raw rows: {len(df_raw_case1)}")

    # -------- CASE3 BOOTSTRAP --------
    state.runner_case3 = ForecastMultiRunner(root_dir=OUT_ROOT)
    state.forecaster_case3 = HorizonForecasterClassic(run_root=FORECASTER_ROOT)
    state.df_raw_case3 = load_raw_timeseries(DATA_CSV)
    print(f"[bootstrap] CASE3 df_raw rows: {len(state.df_raw_case3)}")

    # -------- CASE2 BOOTSTRAP --------
    # kalau case2 perlu data awal, load di sini juga (nanti)
    print("[bootstrap] CASE2 ready (placeholder)")

# -------------------------------------------------
# Request / Response models
# -------------------------------------------------

class InferRequest(BaseModel):
    query: str

class InferResponse(BaseModel):
    raw: Dict[str, Any]
    llmContext: Dict[str, Any]

# -------------------------------------------------
# Helper builder llmContext untuk masing-masing case
# -------------------------------------------------

def build_llm_context_case1(full_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    case1_path_pipeline.run_query() output kamu tadi bentuknya:
      {
        "human_report": "...kalimat ringkas untuk operator...",
        "intent": {...},
        "llm_metadata": {...},
        "plot_paths": [...]
      }

    Kita ekstrak yang aman dan useful.
    """
    return {
        "intent": full_result.get("intent"),
        "summary_brief": full_result.get("human_report"),
        "metrics": full_result.get("llm_metadata"),
        # jangan kirim plot_paths mentah ke LLM, tapi boleh infonya
        "has_plots": bool(full_result.get("plot_paths")),
    }

def build_llm_context_case2(full_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder buat case2. Nanti kamu ubah sesuai output nyata case2.
    Misal case2 itu produksi / R&D request / bahan kurang, dll.
    """
    return {
        "intent": full_result.get("intent", {"kind": "CASE2_PLACEHOLDER"}),
        "summary_brief": full_result.get("summary", "no summary"),
        "details": full_result.get("details", []),
    }

def build_llm_context_case3(full_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mirip yang kita bahas tadi: forecast, ETA, anomaly, threshold.
    """
    intent = full_result.get("intent", {})
    forecast_info = full_result.get("forecast", {})
    eta_info = full_result.get("eta", {})
    anomalies = full_result.get("anomalies", [])

    llm_ctx = {
        "intent": intent,
        "summary": {
            "hasAnomaly": len(anomalies) > 0,
            "affectedMachines": list({a.get("machine_id") for a in anomalies}),
        },
        "eta": eta_info,
        "forecast_keypoints": forecast_info.get("keypoints")
            if isinstance(forecast_info, dict)
            else None,
        "threshold_notes": full_result.get("threshold_notes"),
        "operator_briefing": full_result.get("operator_briefing"),
    }
    return llm_ctx

# -------------------------------------------------
# Endpoint per case
# -------------------------------------------------

@app.post("/infer_case1", response_model=InferResponse)
def infer_case1(req: InferRequest):
    """
    Pertanyaan historis/tren/statistik -> case1_path_pipeline.run_query
    """
    q = req.query
    result_case1 = run_query_case1(
        q,
        state.df_raw_case1,
        plots_dir="./plots"  # sesuai yang di contoh kamu
    )

    return {
        "raw": result_case1,
        "llmContext": build_llm_context_case1(result_case1),
    }


@app.post("/infer_case2", response_model=InferResponse)
def infer_case2(req: InferRequest):
    """
    Placeholder untuk logic case2 kamu.
    Nanti ganti run_full_query_case2 dengan pipeline sebenarnya.
    """
    q = req.query
    result_case2 = run_full_query_case2(q)

    return {
        "raw": result_case2,
        "llmContext": build_llm_context_case2(result_case2),
    }


@app.post("/infer_case3", response_model=InferResponse)
def infer_case3(req: InferRequest):
    """
    Pertanyaan forecast/anomali/ETA (turun di bawah threshold jam berapa, dsb)
    """
    q = req.query
    result_case3 = run_full_query_case3(
        user_text=q,
        runner=state.runner_case3,
        df_raw=state.df_raw_case3,
        forecaster=state.forecaster_case3,
        n_lags=3,
    )

    return {
        "raw": result_case3,
        "llmContext": build_llm_context_case3(result_case3),
    }
