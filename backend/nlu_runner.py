# nlu_runner.py
from __future__ import annotations
import os
import json
from typing import Dict, Any, Tuple, List

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import safetensors.torch as st

from .config import BASE_MODEL  # BASE_MODEL kita simpan di config.py
from .trainer_heads import BertRegressor  # reuse arsitektur head regresi


class ForecastMultiRunner:
    """
    Loader + inference untuk semua head NLU:
    - intent
    - scope
    - timecombo (time_granularity|anchor_time_type|time_complexity|horizon_type)
    - horizon regression
    - threshold_metric / threshold_operator / threshold_value
    - whatif_flag
    """

    def __init__(self, root_dir: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # multiclass heads
        self.intent_tok, self.intent_m, self.intent_i2l = self._load_mc(
            os.path.join(root_dir, "intent", "final_model")
        )
        self.scope_tok, self.scope_m, self.scope_i2l = self._load_mc(
            os.path.join(root_dir, "scope", "final_model")
        )
        self.timec_tok, self.timec_m, self.timec_i2l = self._load_mc(
            os.path.join(root_dir, "timecombo", "final_model")
        )
        self.wif_tok, self.wif_m, self.wif_i2l = self._load_mc(
            os.path.join(root_dir, "whatif_flag", "final_model")
        )

        self.thmet_tok, self.thmet_m, self.thmet_i2l = self._load_mc(
            os.path.join(root_dir, "threshold_metric", "final_model")
        )
        self.thop_tok, self.thop_m, self.thop_i2l = self._load_mc(
            os.path.join(root_dir, "threshold_operator", "final_model")
        )

        # regression heads
        self.hor_tok, self.hor_m, self.hor_ykeys = self._load_reg(
            os.path.join(root_dir, "horizon", "final_model")
        )
        self.thval_tok, self.thval_m, self.thval_y = self._load_reg(
            os.path.join(root_dir, "threshold_value", "final_model")
        )

    # ---------- internal loaders ----------

    def _load_mc(self, model_dir: str):
        """
        Load 1 head klasifikasi multilabel (intent, scope, dst.)
        Returns: tokenizer, model.eval(), id2label
        """
        cfg = AutoConfig.from_pretrained(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = (
            AutoModelForSequenceClassification
            .from_pretrained(model_dir)
            .eval()
            .to(self.device)
        )
        i2l = cfg.id2label
        return tok, mdl, i2l

    def _load_reg(self, model_dir: str):
        """
        Load 1 head regresi horizon / threshold_value.
        Kita recreate BertRegressor(BASE_MODEL, out_dim=len(y_keys)),
        lalu kita isi weight dari safetensors.
        """
        # tokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)

        # target dim (y_keys) disimpan di parent folder lewat target_map.json
        parent_dir = os.path.dirname(model_dir)
        with open(os.path.join(parent_dir, "target_map.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        y_keys = meta["y_keys"]

        # inisialisasi skeleton
        reg = BertRegressor(BASE_MODEL, out_dim=len(y_keys))

        # load weight dari safetensors kalau ada, fallback ke pytorch_model.bin kalau perlu
        safetensor_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(safetensor_path):
            state_dict = st.load_file(safetensor_path, device="cpu")
        else:
            # fallback legacy
            state_dict = torch.load(
                os.path.join(model_dir, "pytorch_model.bin"),
                map_location="cpu"
            )
        reg.load_state_dict(state_dict)

        reg.eval().to(self.device)

        return tok, reg, y_keys

    # ---------- low-level run helpers ----------

    @staticmethod
    def _decode_label(i2l, idx: int) -> str:
        """
        Convert index → label string dari id2label yang mungkin dict {0:"A",1:"B"...}
        atau {"0":"A",...} atau list.
        """
        if isinstance(i2l, dict):
            return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
        try:
            return i2l[idx]
        except Exception:
            return f"label_{idx}"

    @torch.no_grad()
    def _run_mc(self, tok, mdl, i2l, text: str, max_len: int = 128):
        enc = tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        logits = mdl(**enc).logits.squeeze(0)
        probs = torch.softmax(logits, dim=0)

        idx = int(torch.argmax(probs).item())
        lab = self._decode_label(i2l, idx)
        conf = float(probs[idx].item())

        return lab, conf

    @torch.no_grad()
    def _run_reg(self, tok, mdl, ykeys, text: str, max_len: int = 128):
        enc = tok(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        out = mdl(**enc)
        preds = out["logits"].squeeze(0).cpu().numpy().tolist()

        return {
            ykeys[i]: float(preds[i])
            for i in range(len(ykeys))
        }

    # ---------- public API ----------

    def predict(self, text: str, max_len: int = 128) -> Dict[str, Any]:
        """
        Jalankan semua head sekaligus dan balikin struktur kaya:
        {
          "intent": {...},
          "scope": {...},
          "time": {...},
          "horizon_minutes": {...},
          "threshold": {...},
          "what_if": {...},
          "route_hints": [...]
        }
        """

        # 1. intent
        intent_lab, intent_conf = self._run_mc(
            self.intent_tok, self.intent_m, self.intent_i2l, text, max_len
        )

        # 2. scope
        scope_lab, scope_conf = self._run_mc(
            self.scope_tok, self.scope_m, self.scope_i2l, text, max_len
        )

        # 3. time combo
        timec_lab, timec_conf = self._run_mc(
            self.timec_tok, self.timec_m, self.timec_i2l, text, max_len
        )
        # timecombo label = "gran|anchor|complexity|horizonType"
        (
            time_granularity,
            anchor_time_type,
            time_complexity,
            horizon_type,
        ) = timec_lab.split("|")

        # 4. horizon regression
        horizon_pred = self._run_reg(
            self.hor_tok, self.hor_m, self.hor_ykeys, text, max_len
        )

        # 5. threshold classification + value regression
        thr_metric_lab, thr_metric_conf = self._run_mc(
            self.thmet_tok, self.thmet_m, self.thmet_i2l, text, max_len
        )
        thr_oper_lab, thr_oper_conf = self._run_mc(
            self.thop_tok, self.thop_m, self.thop_i2l, text, max_len
        )
        thr_value_pred = self._run_reg(
            self.thval_tok, self.thval_m, self.thval_y, text, max_len
        )

        # 6. what-if flag
        whatif_lab, whatif_conf = self._run_mc(
            self.wif_tok, self.wif_m, self.wif_i2l, text, max_len
        )
        is_whatif = (whatif_lab == "IS_WHAT_IF")

        # 7. routing hints downstream
        route = []
        if scope_lab == "ALL":
            route.append("aggregate_all_machines")
        else:
            route.append("machine_entity_extractor")

        if intent_lab == "FORECAST_VALUE":
            route.append("forecast_numeric_pipeline")
        elif intent_lab == "FORECAST_TREND":
            route.append("forecast_trend_pipeline")
        elif intent_lab == "THRESHOLD_TIME":
            route.append("threshold_time_pipeline")
        elif intent_lab == "WHAT_IF_FORECAST":
            route.append("counterfactual_simulation_pipeline")

        route.append(
            f"time_parse_{time_granularity.lower()}_{time_complexity.lower()}"
        )

        if is_whatif:
            route.append("inject_counterfactual_state")

        return {
            "intent": {
                "label": intent_lab,
                "confidence": round(intent_conf, 4),
            },
            "scope": {
                "label": scope_lab,
                "confidence": round(scope_conf, 4),
            },
            "time": {
                "time_granularity": time_granularity,
                "anchor_time_type": anchor_time_type,
                "time_complexity": time_complexity,
                "horizon_type": horizon_type,
                "confidence": round(timec_conf, 4),
            },
            "horizon_minutes": horizon_pred,  # ex: {"y_start": ~3.0, "y_end": ~48.0}
            "threshold": {
                "metric": {
                    "label": thr_metric_lab,
                    "confidence": round(thr_metric_conf, 4),
                },
                "operator": {
                    "label": thr_oper_lab,
                    "confidence": round(thr_oper_conf, 4),
                },
                "value_predicted": thr_value_pred,  # ex: {"y_val": ~80.0}
            },
            "what_if": {
                "flag": whatif_lab,
                "is_counterfactual": is_whatif,
                "confidence": round(whatif_conf, 4),
            },
            "route_hints": route,
        }
