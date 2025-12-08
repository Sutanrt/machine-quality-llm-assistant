# ============================================================
# inference_demo.py
#
# Tujuan:
# - Load model 4-head (AnomalyQuadRunner)
# - Load data sensor df_raw2.csv
# - Define threshold min/max per mesin
# - Jalankan query contoh (bahasa natural)
# - Tampilkan hasil anomaly secara ringkas
#
# Catatan:
# - Ini pure inference/demo. Tidak ada training.
# ============================================================

from __future__ import annotations
import os
import torch
import pandas as pd

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# --- module internal yang sudah kita punya dari modul lain ---
from query_interpret import smart_predict
from time_parse import resolve_time_window
from anomaly_query_engine import run_anomaly_query
from other_ts_loader import load_timeseries_from_df_raw2
from payload_formatter import anomaly_result_to_payload, pretty_print_payload


# ============================================================
# (0) Config dan contoh query
# ============================================================

OUT_ROOT = "./runs_anomaly_quad_relclass"   # folder output model hasil training
TS_CSV_PATH = "/workspace/df_raw2.csv"      # path data sensor bersih

# Contoh pertanyaan user
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

# Dummy thresholds (ganti ini nanti pakai loader threshold sheet Excel)
DF_THRESHOLDS = pd.DataFrame({
    "machine_id": ["XP888A", "PQ667"],
    "max_value":  [10,       140],
    "min_value":  [0,        110],
})


# ============================================================
# (1) Kelas runtime model intent (Quad Head)
#     -> ini sama kayak di kode kamu, cuma kita rapihin sedikit gaya
# ============================================================

class AnomalyQuadRunner:
    """
    Wrapper empat head classifier:
      - scope (ALL / NOT_ALL)
      - time_granularity (HOUR / DAY / WEEK / MONTH / SHIFT / ...)
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

        # Case (A1..A15 + R1..R6)
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
        # id2label dari HF bisa dict str->label atau list index-based
        if isinstance(i2l, dict):
            return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
        try:
            return i2l[idx]
        except Exception:
            return f"label_{idx}"

    @torch.no_grad()
    def predict(self, text: str, max_len=128):
        """
        Infer 4 head model untuk 1 query text.
        Return dict:
          {
            "scope": {"label": ..., "confidence": ...},
            "time_granularity": {...},
            "time_complexity": {...},
            "case": {...},
            "route_hints": [...]
          }
        """
        # scope
        s = self.scope_tok(
            text, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        ).to(self.device)
        s_logits = self.scope_m(**s).logits.squeeze(0).cpu().numpy()
        s_id = int(s_logits.argmax())
        s_lab = self._decode(self.scope_i2l, s_id)
        s_conf = float(torch.softmax(torch.from_numpy(s_logits), dim=0)[s_id])

        # granularity
        g = self.gran_tok(
            text, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        ).to(self.device)
        g_logits = self.gran_m(**g).logits.squeeze(0).cpu().numpy()
        g_id = int(g_logits.argmax())
        g_lab = self._decode(self.gran_i2l, g_id)
        g_conf = float(torch.softmax(torch.from_numpy(g_logits), dim=0)[g_id])

        # complexity
        x = self.cplx_tok(
            text, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        ).to(self.device)
        x_logits = self.cplx_m(**x).logits.squeeze(0).cpu().numpy()
        x_id = int(x_logits.argmax())
        x_lab = self._decode(self.cplx_i2l, x_id)
        x_conf = float(torch.softmax(torch.from_numpy(x_logits), dim=0)[x_id])

        # case
        c = self.case_tok(
            text, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        ).to(self.device)
        c_logits = self.case_m(**c).logits.squeeze(0).cpu().numpy()
        c_id = int(c_logits.argmax())
        c_lab = self._decode(self.case_i2l, c_id)
        c_conf = float(torch.softmax(torch.from_numpy(c_logits), dim=0)[c_id])

        # routing hints awal
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
# (2) Fungsi helper kecil
# ============================================================

def build_runner() -> AnomalyQuadRunner:
    """
    Load semua head dari folder OUT_ROOT.
    """
    return AnomalyQuadRunner(
        scope_dir=os.path.join(OUT_ROOT, "scope", "final_model"),
        gran_dir=os.path.join(OUT_ROOT, "time_granularity", "final_model"),
        cplx_dir=os.path.join(OUT_ROOT, "time_complexity", "final_model"),
        case_dir=os.path.join(OUT_ROOT, "case", "final_model"),
    )


def run_one_query(q: str, runner: AnomalyQuadRunner, df_ts: pd.DataFrame, df_thr: pd.DataFrame):
    """
    Jalankan full pipeline utk satu pertanyaan q:
    - smart_predict (postprocess rules)
    - resolve_time_window (jadi start/end konkret)
    - run_anomaly_query (filter data & cari out-of-range)
    - anomaly_result_to_payload (rapikan output)
    - pretty_print_payload (print human readable)
    """
    print("\nQ:", q)

    # 1. prediksi intent+granularity+scope dll (fix via rules)
    pred_aug = smart_predict(q, runner)
    print("   [smart_predict]:", pred_aug)

    # 2. time window konkret dari teks
    tw = resolve_time_window(q, pred_aug)
    print(f"   [resolved_time_window]: {tw.start} → {tw.end} ({tw.kind}/{tw.granularity})")

    # 3. anomaly engine
    res = run_anomaly_query(
        text=q,
        runner=runner,
        df_ts=df_ts,
        df_thr=df_thr,
        # NOTE:
        #  kalau kamu sudah patch run_anomaly_query dgn pred_override / resolve_time_window_fn,
        #  ganti call di atas jadi:
        # res = run_anomaly_query(
        #     text=q,
        #     runner=runner,
        #     df_ts=df_ts,
        #     df_thr=df_thr,
        #     pred_override=pred_aug,
        #     resolve_time_window_fn=lambda _text, _pred: tw,
        # )
    )

    # 4. bungkus payload final
    payload = anomaly_result_to_payload(res, df_ts)

    # 5. tampilkan ringkas nice
    pretty_print_payload(payload)


# ============================================================
# (3) Main demo
# ============================================================

if __name__ == "__main__":
    print("=== BOOTSTRAP MODEL & DATA ===")
    runner = build_runner()
    df_ts  = load_timeseries_from_df_raw2(TS_CSV_PATH)
    df_thr = DF_THRESHOLDS.copy()

    print("\n=== DEMO PREDIKSI ===")
    for q in EXAMPLES:
        try:
            run_one_query(q, runner, df_ts, df_thr)
        except Exception as e:
            print("   [ERROR]:", e)

    print("\nModel tersimpan di:", OUT_ROOT)
