# model_runner.py
# Loader inference untuk triple-head classifier.
# File ini aman diimport dari main_pipeline.py

import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class DualTripleRunner:
    """
    Wrapper 3-head:
      - scope: ALL vs NOT_ALL
      - time:  HOUR/DAY/WEEK/MONTH/YEAR/ALL
      - case:  SINGLE_PERIOD / COMPARE / RANGE_... / SUMMARY / ...
    Output sesuai format yang dipakai interpret().
    """

    def __init__(self, scope_model_dir: str, time_model_dir: str, case_model_dir: str, device: str = None):
        # pilih device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Scope head
        self.scope_tok = AutoTokenizer.from_pretrained(scope_model_dir)
        self.scope_cfg = AutoConfig.from_pretrained(scope_model_dir)
        self.scope_i2l = self.scope_cfg.id2label
        self.scope_m = AutoModelForSequenceClassification.from_pretrained(scope_model_dir).to(self.device).eval()

        # Time head
        self.time_tok = AutoTokenizer.from_pretrained(time_model_dir)
        self.time_cfg = AutoConfig.from_pretrained(time_model_dir)
        self.time_i2l = self.time_cfg.id2label
        self.time_m = AutoModelForSequenceClassification.from_pretrained(time_model_dir).to(self.device).eval()

        # Case head
        self.case_tok = AutoTokenizer.from_pretrained(case_model_dir)
        self.case_cfg = AutoConfig.from_pretrained(case_model_dir)
        self.case_i2l = self.case_cfg.id2label
        self.case_m = AutoModelForSequenceClassification.from_pretrained(case_model_dir).to(self.device).eval()

    @staticmethod
    def _decode(i2l, idx: int):
        # i2l bisa dict {0:"ALL",1:"NOT_ALL"} / mapping string-key dsb
        if isinstance(i2l, dict):
            return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
        try:
            return i2l[idx]
        except Exception:
            return f"label_{idx}"

    @torch.no_grad()
    def predict(self, text: str, max_len: int = 128):
        # ---- Scope head ----
        s = self.scope_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        s_logits = self.scope_m(**s).logits.squeeze(0)
        s_probs = torch.softmax(s_logits, dim=0)
        s_id = int(torch.argmax(s_probs).item())
        s_label = self._decode(self.scope_i2l, s_id)
        s_conf = float(s_probs[s_id].item())

        # ---- Time head ----
        t = self.time_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        t_logits = self.time_m(**t).logits.squeeze(0)
        t_probs = torch.softmax(t_logits, dim=0)
        t_id = int(torch.argmax(t_probs).item())
        t_label = self._decode(self.time_i2l, t_id)
        t_conf = float(t_probs[t_id].item())

        # ---- Case head ----
        c = self.case_tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        c_logits = self.case_m(**c).logits.squeeze(0)
        c_probs = torch.softmax(c_logits, dim=0)
        c_id = int(torch.argmax(c_probs).item())
        c_label = self._decode(self.case_i2l, c_id)
        c_conf = float(c_probs[c_id].item())

        # route_hints (opsional, useful downstream)
        route = []
        if c_label.startswith("RANGE"):
            route.append("use_range_extractor")
        if t_label in {"HOUR", "DAY", "WEEK", "MONTH", "YEAR"}:
            route.append(f"slot_{t_label.lower()}_extractor")
        if s_label == "ALL":
            route.append("aggregate_all_machines")
        else:
            route.append("machine_entity_extractor")

        return {
            "scope": {"label": s_label, "confidence": round(s_conf, 4)},
            "time":  {"label": t_label, "confidence": round(t_conf, 4)},
            "case":  {"label": c_label, "confidence": round(c_conf, 4)},
            "route_hints": route,
        }


# ==================================================
# GLOBAL RUNNER INSTANCE
# ==================================================
#
# Ini diasumsikan model sudah dilatih pakai trainer.py
# dan disimpan di:
#   runs_C_triple/scope/final_model
#   runs_C_triple/time/final_model
#   runs_C_triple/case/final_model
#
# Kamu bisa ubah path ini kalau lokasi model beda.

_DEFAULT_SCOPE_DIR = "/workspace/runs_C_triple/scope/final_model"
_DEFAULT_TIME_DIR  = "/workspace/runs_C_triple/time/final_model"
_DEFAULT_CASE_DIR  = "/workspace/runs_C_triple/case/final_model"

runner = DualTripleRunner(
    scope_model_dir=_DEFAULT_SCOPE_DIR,
    time_model_dir=_DEFAULT_TIME_DIR,
    case_model_dir=_DEFAULT_CASE_DIR,
)
