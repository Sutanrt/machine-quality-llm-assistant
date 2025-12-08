# ============================================================
# pipeline_full.py
#
# End-to-end anomaly QA pipeline:
#   1. user query (bahasa natural)
#   2. intent classification (4-head model + smart rules)
#   3. time window resolution
#   4. anomaly scan (bandingkan data ts vs threshold)
#   5. payload siap buat LLM / UI dashboard
#
# Cocok buat:
# - smoke test manual di Colab
# - nanti di-wrap jadi API FastAPI/Flask
# ============================================================

from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import os
import torch
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
MODEL_ROOT   = "./runs_anomaly_quad_relclass"  # folder berisi 4 head final_model
TS_CSV_PATH  = "/workspace/df_raw2.csv"        # hasil extractor df_raw2.csv
# TODO prod: load threshold dari Excel pakai loader threshold
DF_THRESHOLDS = pd.DataFrame({
    "machine_id": ["XP888A", "PQ667"],
    "max_value":  [10,       140],
    "min_value":  [0,        110],
})

# -----------------------------
# IMPORT MODUL INTERNAL
# -----------------------------
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from query_interpret import smart_predict            # post-process intent
from time_parse import resolve_time_window           # text+pred -> TimeWindow(start,end,...)
from anomaly_query_engine import run_anomaly_query   # orchestrator text→AnomalyResult
from other_ts_loader import load_timeseries_from_df_raw2
from payload_formatter import (
    anomaly_result_to_payload,
    pretty_print_payload,
)

# ============================================================
# (1) Runtime model intent (Quad Head)
#    -> diambil dari versi kamu yang asli (:contentReference[oaicite:0]{index=0})
# ============================================================

class AnomalyQuadRunner:
    """
    Wrapper empat head classifier:
      - scope (ALL / NOT_ALL)
      - time_granularity (HOUR / DAY / WEEK / MONTH / SHIFT / YEAR / ...)
      - time_complexity (RANGE_CLEAR / RANGE_REL / PERIOD / ...)
      - case (A1..A15 / R1..R6)
    """
    def __init__(self, scope_dir, gran_dir, cplx_dir, case_dir):
        # Scope
        self.scope_tok = AutoTokenizer.from_pretrained(scope_dir)
        self.scope_cfg = AutoConfig.from_pretrained(scope_dir)
        self.scope_i2l = self.scope_cfg.id2label
        self.scope_m   = AutoModelForSequenceClassification.from_pretrained(scope_dir).eval()

        # Granularity
        self.gran_tok = AutoTokenizer.from_pretrained(gran_dir)
        self.gran_cfg = AutoConfig.from_pretrained(gran_dir)
        self.gran_i2l = self.gran_cfg.id2label
        self.gran_m   = AutoModelForSequenceClassification.from_pretrained(gran_dir).eval()

        # Complexity
        self.cplx_tok = AutoTokenizer.from_pretrained(cplx_dir)
        self.cplx_cfg = AutoConfig.from_pretrained(cplx_dir)
        self.cplx_i2l = self.cplx_cfg.id2label
        self.cplx_m   = AutoModelForSequenceClassification.from_pretrained(cplx_dir).eval()

        # Case
        self.case_tok = AutoTokenizer.from_pretrained(case_dir)
        self.case_cfg = AutoConfig.from_pretrained(case_dir)
        self.case_i2l = self.case_cfg.id2label
        self.case_m   = AutoModelForSequenceClassification.from_pretrained(case_dir).eval()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in [self.scope_m, self.gran_m, self.cplx_m, self.case_m]:
            m.to(self.device)

    @staticmethod
    def _decode(i2l, idx: int):
        # HuggingFace id2label bisa dict {0:'ALL',1:'NOT_ALL',...} atau list
        if isinstance(i2l, dict):
            return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
        try:
            return i2l[idx]
        except Exception:
            return f"label_{idx}"

    @torch.no_grad()
    def predict(self, text: str, max_len=128):
        # scope
        s = self.scope_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(self.device)
        s_logits = self.scope_m(**s).logits.squeeze(0).cpu().numpy()
        s_id = int(s_logits.argmax())
        s_lab = self._decode(self.scope_i2l, s_id)
        s_conf = float(torch.softmax(torch.from_numpy(s_logits), dim=0)[s_id])

        # granularity
        g = self.gran_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(self.device)
        g_logits = self.gran_m(**g).logits.squeeze(0).cpu().numpy()
        g_id = int(g_logits.argmax())
        g_lab = self._decode(self.gran_i2l, g_id)
        g_conf = float(torch.softmax(torch.from_numpy(g_logits), dim=0)[g_id])

        # complexity
        x = self.cplx_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(self.device)
        x_logits = self.cplx_m(**x).logits.squeeze(0).cpu().numpy()
        x_id = int(x_logits.argmax())
        x_lab = self._decode(self.cplx_i2l, x_id)
        x_conf = float(torch.softmax(torch.from_numpy(x_logits), dim=0)[x_id])

        # case
        c = self.case_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(self.device)
        c_logits = self.case_m(**c).logits.squeeze(0).cpu().numpy()
        c_id = int(c_logits.argmax())
        c_lab = self._decode(self.case_i2l, c_id)
        c_conf = float(torch.softmax(torch.from_numpy(c_logits), dim=0)[c_id])

        # route hints awal buat downstream
        route = []
        route.append("aggregate_all_machines" if s_lab == "ALL" else "machine_entity_extractor")
        if g_lab in {"HOUR", "DAY", "WEEK", "MONTH", "YEAR", "SHIFT"}:
            route.append(f"slot_{g_lab.lower()}_extractor")
        if x_lab in {"RANGE_CLEAR", "RANGE_REL", "WEEK_N", "POINT", "PERIOD", "SHIFT"}:
            route.append(f"time_complexity_{x_lab.lower()}")

        return {
            "scope": {"label": s_lab, "confidence": round(s_conf, 4)},
            "time_granularity": {"label": g_lab, "confidence": round(g_conf, 4)},
            "time_complexity": {"label": x_lab, "confidence": round(x_conf, 4)},
            "case": {"label": c_lab, "confidence": round(c_conf, 4)},
            "route_hints": route,
        }


# ============================================================
# (2) Bootstrap util
# ============================================================

def build_runner(model_root: str) -> AnomalyQuadRunner:
    """
    Load semua head classifier dari folder model_root.
    """
    return AnomalyQuadRunner(
        scope_dir=os.path.join(model_root, "scope", "final_model"),
        gran_dir=os.path.join(model_root, "time_granularity", "final_model"),
        cplx_dir=os.path.join(model_root, "time_complexity", "final_model"),
        case_dir=os.path.join(model_root, "case", "final_model"),
    )

def load_sensor_timeseries(ts_csv_path: str) -> pd.DataFrame:
    """
    Load dataframe sensor final (machine_id, ts[datetime64], value[float])
    dari df_raw2.csv yang udah dibersihin sebelumnya.
    """
    # asumsi timestamp historis = MM/DD/YYYY HH:MM
    # kalau ternyata DD/MM/YYYY ubah argumen assume_us_datetime=False
    return load_timeseries_from_df_raw2(
        ts_csv_path,
        assume_us_datetime=True,
        verbose=False,
    )


# ============================================================
# (3) Core: jalankan 1 query user end-to-end
# ============================================================

def run_query_once(
    query_text: str,
    runner: AnomalyQuadRunner,
    df_ts: pd.DataFrame,
    df_thr: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Satu kalimat user → payload anomaly JSON-friendly.

    Step:
    1. smart_predict() → intent final (scope, granularity, complexity, case, route_hints, dll)
    2. resolve_time_window() → TimeWindow(start, end, ...)
    3. run_anomaly_query() → cari anomali di df_ts sesuai window + threshold
    4. anomaly_result_to_payload() → siap kirim ke UI / LLM
    """

    # 1. intent + fix rule (scope SUBSET / ALL, granularity SHIFT/HOUR/WEEK/etc, mapping case A*/R*)
    pred_aug = smart_predict(query_text, runner)

    # 2. interpret waktu (hari ini? minggu lalu? shift malam? 1–20 Jan 2025? dst)
    tw = resolve_time_window(query_text, pred_aug)

    # 3. cari anomaly di window tsb
    #    NOTE: kita pakai versi run_anomaly_query yang SUDAH dipatch
    #    untuk dukung pred_override dan resolve_time_window_fn.
    result = run_anomaly_query(
        text=query_text,
        runner=runner,
        df_ts=df_ts,
        df_thr=df_thr,
        pred_override=pred_aug,
        resolve_time_window_fn=lambda _txt, _pred: tw,
    )

    # 4. bungkus jadi payload json-friendly
    payload = anomaly_result_to_payload(result, df_ts)
    return payload


# ============================================================
# (4) Demo batch (smoke test)
# ============================================================

EXAMPLES = [
    "Apakah ada anomali mesin XP888A dan PQ667 7 hari terakhir?",
    "Tolong tampilkan anomali semua mesin bulan ini",
    "Cek anomali mesin XP888A tanggal 1–20 Januari 2025",
    "Cek anomali mesin XP888A tanggal 1–20 Januari dan februari 2025",
    "Cek anomali semua mesin bulan Januari dan februari 2025",
    "Cek anomali semua mesin bulan Januari 2025",
    "Cek anomali semua mesin bulan februari 2025",
    "Ada anomali semua mesin minggu lalu?",
    "Periksa anomali shift malam untuk XP888A",
    "apakah ada anomali kemarin malam?",
    "apakah ada anomali di senin minggu lalu pada pukul 8 pagi hingga 10 pagi?",
    "dalam rentang 30 menit terakhir apakah ada keanehan pada mesin?",
]

def demo_batch():
    print("=== BOOTSTRAP MODEL & DATA ===")
    runner = build_runner(MODEL_ROOT)
    df_ts  = load_sensor_timeseries(TS_CSV_PATH)
    df_thr = DF_THRESHOLDS.copy()

    print("\n=== DEMO QUERY ===")
    for q in EXAMPLES:
        print("\n-------------------------------------------------")
        print("QUERY:", q)
        try:
            payload = run_query_once(
                query_text=q,
                runner=runner,
                df_ts=df_ts,
                df_thr=df_thr,
            )
            # tampilkan ringkas human-readable
            pretty_print_payload(payload)
        except Exception as e:
            print("ERROR:", e)

    print("\n[INFO]")
    print("Model root      :", MODEL_ROOT)
    print("Sensor rows     :", len(df_ts))
    print("Threshold rows  :", len(df_thr))


# ============================================================
# (5) Entry point
# ============================================================

if __name__ == "__main__":
    demo_batch()
