# =============================================================
# intent_runtime.py
# Lightweight runtime inference for the 4-head intent model.
# This is what production / pipeline will import.
# =============================================================

from __future__ import annotations
from typing import Dict, Any, List
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class AnomalyQuadRunner:
    """
    Gabungan 4 classifier:
      - scope
      - time_granularity
      - time_complexity
      - case

    Masing-masing head di-train terpisah lalu disimpan di:
      <OUT_ROOT>/<head>/final_model/
    Contoh:
      scope_dir=./runs_anomaly_quad_relclass/scope/final_model
    """

    def __init__(self, scope_dir: str, gran_dir: str, cplx_dir: str, case_dir: str):
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

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in [self.scope_m, self.gran_m, self.cplx_m, self.case_m]:
            m.to(self.device)

    @staticmethod
    def _decode(i2l, idx: int) -> str:
        """
        Model HF simpan label map di config.id2label.
        Bentuknya bisa {0:"ALL",1:"SUBSET"} atau list.
        Kita bikin robust.
        """
        if isinstance(i2l, dict):
            return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
        try:
            return i2l[idx]
        except Exception:
            return f"label_{idx}"

    @torch.no_grad()
    def _predict_single_head(self, tok, model, i2l, text: str, max_len: int = 128):
        batch = tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        logits = model(**batch).logits.squeeze(0).cpu().numpy()
        best_id = int(logits.argmax())
        best_label = self._decode(i2l, best_id)

        # confidence pakai softmax
        probs = torch.softmax(torch.from_numpy(logits), dim=0)
        conf = float(probs[best_id])

        return best_label, conf

    @torch.no_grad()
    def predict(self, text: str, max_len: int = 128) -> Dict[str, Any]:
        """
        Return struktur siap pakai downstream:
        {
          "scope": {...},
          "time_granularity": {...},
          "time_complexity": {...},
          "case": {...},
          "route_hints": [...]
        }
        """
        # scope
        scope_label, scope_conf = self._predict_single_head(
            self.scope_tok, self.scope_m, self.scope_i2l, text, max_len
        )

        # granularity
        gran_label, gran_conf = self._predict_single_head(
            self.gran_tok, self.gran_m, self.gran_i2l, text, max_len
        )

        # complexity
        cplx_label, cplx_conf = self._predict_single_head(
            self.cplx_tok, self.cplx_m, self.cplx_i2l, text, max_len
        )

        # case
        case_label, case_conf = self._predict_single_head(
            self.case_tok, self.case_m, self.case_i2l, text, max_len
        )

        # routing hints buat downstream pipeline
        route: List[str] = []
        route.append(
            "aggregate_all_machines" if scope_label == "ALL" else "machine_entity_extractor"
        )
        if gran_label in {"HOUR","DAY","WEEK","MONTH","YEAR","SHIFT"}:
            route.append(f"slot_{gran_label.lower()}_extractor")
        if cplx_label in {"RANGE_CLEAR","RANGE_REL","WEEK_N","POINT","PERIOD","SHIFT"}:
            route.append(f"time_complexity_{cplx_label.lower()}")

        return {
            "scope": {"label": scope_label, "confidence": round(scope_conf, 4)},
            "time_granularity": {"label": gran_label, "confidence": round(gran_conf, 4)},
            "time_complexity": {"label": cplx_label, "confidence": round(cplx_conf, 4)},
            "case": {"label": case_label, "confidence": round(case_conf, 4)},
            "route_hints": route,
        }


if __name__ == "__main__":
    # contoh pemakaian lokal / sanity check
    OUT_ROOT = "./runs_anomaly_quad_relclass"

    runner = AnomalyQuadRunner(
        scope_dir = f"{OUT_ROOT}/scope/final_model",
        gran_dir  = f"{OUT_ROOT}/time_granularity/final_model",
        cplx_dir  = f"{OUT_ROOT}/time_complexity/final_model",
        case_dir  = f"{OUT_ROOT}/case/final_model",
    )

    examples = [
        "Apakah ada anomali mesin XP888A dan PQ667 7 hari terakhir?",
        "Tolong tampilkan anomali semua mesin bulan ini",
        "Cek anomali mesin XP888A tanggal 10–20 Januari 2025",
        "Ada anomali semua mesin minggu lalu?",
        "Periksa anomali shift malam untuk XP888A",
    ]
    for q in examples:
        print("\nQ:", q)
        print(runner.predict(q))
