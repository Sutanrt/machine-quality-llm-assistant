# interpret_query.py
from __future__ import annotations
import re
from typing import Dict, Any, List

# TODO: nanti pindahin ini ke config.py juga supaya gampang update daftar mesin
KNOWN_MACHINES = ["XP888A","PQ667","ZT901B","LM442","KJ3-A","ALPHA01","LINE-7"]


def _extract_machines(user_text: str, scope_label: str) -> List[str]:
    """
    Cari referensi mesin dari teks user.
    - scope ALL  -> ["ALL"]
    - mesin disebut eksplisit -> list unik
    - tidak disebut (tapi scope SINGLE/MULTI) -> ["UNKNOWN"]
    """
    if scope_label == "ALL":
        return ["ALL"]

    found = []
    for m in KNOWN_MACHINES:
        pattern = r"\b" + re.escape(m) + r"\b"
        if re.search(pattern, user_text, flags=re.IGNORECASE):
            found.append(m)

    # dedup sambil preserve urutan
    found = list(dict.fromkeys(found))
    return found if found else ["UNKNOWN"]


def _snap_single_point(start_pred: float, end_pred: float):
    """
    Model horizon regression kadang ngasih y_start != y_end
    padahal time_complexity = SINGLE_POINT.
    Kita ambil rata-rata lalu bulatkan.
    """
    avg_val = (float(start_pred) + float(end_pred)) / 2.0
    avg_val = round(avg_val)
    return avg_val, avg_val


def _parse_explicit_horizon(user_text: str, time_info: Dict[str, Any]):
    """
    Hard-rule extractor buat horizon.
    Return (start_min, end_min, used_rule: bool)

    Contoh:
    - "2 jam lagi"
    - "30 menit ke depan"
    - "dalam 15 menit"
    """
    tcomp = time_info["time_complexity"]
    txt = user_text.lower()

    def _to_int(m):
        try:
            return int(m.group(1))
        except Exception:
            return None

    # "<N> menit" / "mnt"
    m_menit = re.search(r"(\d+)\s*(menit|mnt)\b", txt)
    if m_menit:
        n_min = _to_int(m_menit)
        if n_min is not None:
            if (
                "ke depan" in txt
                or "kedepan" in txt
                or "sampai" in txt
                or "interval" in txt
            ):
                # RANGE 0..N
                return 0, n_min, True
            # fallback
            if tcomp == "SINGLE_POINT":
                return n_min, n_min, True
            return 0, n_min, True

    # "<N> jam" / "jm"
    m_jam = re.search(r"(\d+)\s*(jam|jm)\b", txt)
    if m_jam:
        n_hour = _to_int(m_jam)
        if n_hour is not None:
            total_min = n_hour * 60
            if tcomp == "SINGLE_POINT":
                return total_min, total_min, True
            return 0, total_min, True

    # "dalam <N> menit ke depan"
    m_dlm = re.search(r"dalam\s+(\d+)\s*(menit|mnt)", txt)
    if m_dlm:
        n_min = int(m_dlm.group(1))
        return 0, n_min, True

    # "<N> menit lagi/depan/kedepan"
    m_simple_m = re.search(r"(\d+)\s*(menit|mnt)\s*(lagi|depan|kedepan)", txt)
    if m_simple_m:
        n_min = int(m_simple_m.group(1))
        if tcomp == "SINGLE_POINT":
            return n_min, n_min, True
        return 0, n_min, True

    # "<N> jam lagi/depan/kedepan"
    m_simple_h = re.search(r"(\d+)\s*(jam|jm)\s*(lagi|depan|kedepan)", txt)
    if m_simple_h:
        n_h = int(m_simple_h.group(1))
        tot = n_h * 60
        if tcomp == "SINGLE_POINT":
            return tot, tot, True
        return 0, tot, True

    return None, None, False


def extract_all_requested_minutes(user_text: str):
    """
    Cari semua horizon eksplisit multi-point, contoh:
    "48 menit, 24 menit, 12 menit"
    "2 jam, 1 jam"
    Output: list unik menit (sorted).
    """
    pattern = r"(\d+)\s*(menit|mnt|minute|min|jam|hour|h)"
    hits = re.findall(pattern, user_text.lower())

    mins = []
    for num_str, unit in hits:
        val = int(num_str)
        if "jam" in unit or unit in ["hour", "h"]:
            mins.append(val * 60)
        else:
            mins.append(val)

    return sorted(list({m for m in mins}))


def interpret_query(user_text: str, runner_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ambil output NLU (ForecastMultiRunner.predict)
    → bangun NormalizedSpec yang dipakai executor.execute_spec()

    NormalizedSpec schema (ringkas):
    {
      action_type: str,
      intent: str,
      scope: str,
      machines: [str],
      time: {
        granularity, anchor_time_type, time_complexity, horizon_type,
        window_minutes: {
          start_min, end_min, source
        },
        multi_targets_minutes: [int] | None
      },
      threshold: {...} | None,
      counterfactual: {
        enabled: bool,
        raw_query: str | None
      },
      route_hints: [...]
    }
    """

    # 1. ambil komponen utama dari runner_output
    intent_lab   = runner_output["intent"]["label"]
    scope_lab    = runner_output["scope"]["label"]
    time_info    = runner_output["time"]
    horizon_pred = runner_output["horizon_minutes"]   # ex: {"y_start":..,"y_end":..}
    whatif_info  = runner_output["what_if"]
    thr_info     = runner_output["threshold"]

    # 2. mesin mana?
    machines = _extract_machines(user_text, scope_lab)

    # 3. tentukan horizon (start_min, end_min)
    explicit_start, explicit_end, ok_rule = _parse_explicit_horizon(
        user_text,
        time_info,
    )

    if ok_rule:
        final_start = explicit_start
        final_end   = explicit_end
        horizon_source = "text_rule"
    else:
        # fallback ke regressor
        y_s = float(horizon_pred.get("y_start", -1))
        y_e = float(horizon_pred.get("y_end", -1))

        if time_info["time_complexity"] == "SINGLE_POINT":
            y_s, y_e = _snap_single_point(y_s, y_e)

        final_start = max(0, round(y_s)) if y_s >= 0 else -1
        final_end   = max(final_start, round(y_e)) if y_e >= 0 else -1
        horizon_source = "model_regressor"

    # 4. threshold (kalau intent THRESHOLD_TIME)
    threshold_spec = None
    if intent_lab == "THRESHOLD_TIME":
        met = thr_info["metric"]["label"]
        op  = thr_info["operator"]["label"]
        val_pred = thr_info["value_predicted"].get("y_val", -1)

        # normalisasi operator → simbol
        if op == "RISE_ABOVE":
            op_symbol = ">"
        elif op == "DROP_BELOW":
            op_symbol = "<"
        else:
            op_symbol = None

        if met != "NONE" and op_symbol is not None and val_pred >= 0:
            threshold_spec = {
                "metric": met,
                "operator": op_symbol,
                "value": round(float(val_pred), 2),
            }

    # 5. counterfactual mode?
    counterfactual_mode = whatif_info["is_counterfactual"]

    # 6. map intent → executor action_type
    if intent_lab == "FORECAST_VALUE":
        action_type = "forecast_numeric"
    elif intent_lab == "FORECAST_TREND":
        action_type = "forecast_trend"
    elif intent_lab == "THRESHOLD_TIME":
        action_type = "threshold_eta"
    elif intent_lab == "WHAT_IF_FORECAST":
        action_type = "counterfactual_forecast"
    else:
        action_type = "unknown"

    # 7. multi horizon (misal "48 menit, 24 menit, 12 menit")
    all_mins = extract_all_requested_minutes(user_text)
    multi_targets = all_mins if len(all_mins) > 1 else None

    # 8. rakit final spec
    spec = {
        "action_type": action_type,
        "intent": intent_lab,
        "scope": scope_lab,
        "machines": machines,
        "time": {
            "granularity": time_info["time_granularity"],
            "anchor_time_type": time_info["anchor_time_type"],
            "time_complexity": time_info["time_complexity"],
            "horizon_type": time_info["horizon_type"],
            "window_minutes": {
                "start_min": final_start,
                "end_min": final_end,
                "source": horizon_source,
            },
            "multi_targets_minutes": multi_targets,
        },
        "threshold": threshold_spec,
        "counterfactual": {
            "enabled": counterfactual_mode,
            "raw_query": user_text if counterfactual_mode else None,
        },
        "route_hints": runner_output["route_hints"],
    }

    return spec


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# import json

# import os, json, random, numpy as np, warnings
# from pathlib import Path
# import torch
# from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error
# from sklearn.model_selection import StratifiedShuffleSplit

# from datasets import load_dataset, Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     AutoConfig,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback
# )
# DATA_DIR   = "/workspace/next"   # ganti sesuai folder dataset hasil generator tadi
# OUT_ROOT   = "/workspace/runs_forecast_heads_v1"
# BASE_MODEL = "indobenchmark/indobert-base-p1" 
# from torch import nn

# class BertRegressor(nn.Module):
#     """
#     Simple regression head on top of base transformer.
#     We'll reuse pretrained model's encoder via AutoModelForSequenceClassification's config trick:
#     Actually easier: load AutoModelForSequenceClassification then replace classifier.
#     BUT we'll just build from AutoModelForSequenceClassification and set num_labels = out_dim,
#     and use MSE loss manually in Trainer via custom compute_loss.
#     """
#     def __init__(self, base_name, out_dim=1):
#         super().__init__()
#         from transformers import AutoModel
#         self.backbone = AutoModel.from_pretrained(base_name)
#         hidden = self.backbone.config.hidden_size
#         self.reg_head = nn.Linear(hidden, out_dim)
#         self.config = self.backbone.config
#         self.out_dim = out_dim

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
#         out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         # use CLS token
#         cls = out.last_hidden_state[:,0,:]
#         preds = self.reg_head(cls)
#         loss = None
#         if labels is not None:
#             loss_fn = nn.MSELoss()
#             loss = loss_fn(preds, labels.float())
#         return {"loss": loss, "logits": preds}

# class ForecastMultiRunner:
#     def __init__(self, root_dir):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # load multiclass heads
#         self.intent_tok, self.intent_m, self.intent_i2l = self._load_mc(os.path.join(root_dir,"intent","final_model"))
#         self.scope_tok, self.scope_m, self.scope_i2l = self._load_mc(os.path.join(root_dir,"scope","final_model"))
#         self.timec_tok, self.timec_m, self.timec_i2l = self._load_mc(os.path.join(root_dir,"timecombo","final_model"))
#         self.wif_tok, self.wif_m, self.wif_i2l       = self._load_mc(os.path.join(root_dir,"whatif_flag","final_model"))

#         self.thmet_tok, self.thmet_m, self.thmet_i2l = self._load_mc(os.path.join(root_dir,"threshold_metric","final_model"))
#         self.thop_tok,  self.thop_m,  self.thop_i2l  = self._load_mc(os.path.join(root_dir,"threshold_operator","final_model"))

#         # load regression heads
#         self.hor_tok, self.hor_m, self.hor_ykeys     = self._load_reg(os.path.join(root_dir,"horizon","final_model"))
#         self.thval_tok, self.thval_m, self.thval_y   = self._load_reg(os.path.join(root_dir,"threshold_value","final_model"))

#     def _load_mc(self, model_dir):
#         cfg = AutoConfig.from_pretrained(model_dir)
#         tok = AutoTokenizer.from_pretrained(model_dir)
#         mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().to(self.device)
#         i2l = cfg.id2label
#         return tok, mdl, i2l

#     # def _load_reg(self, model_dir):
#     #     # load tokenizer
#     #     tok = AutoTokenizer.from_pretrained(model_dir)
#     #     # load custom BertRegressor
#     #     # we re-init same arch used in training:
#     #     reg = BertRegressor(BASE_MODEL, out_dim=len(json.load(open(os.path.join(os.path.dirname(model_dir),"target_map.json")))["y_keys"]))
#     #     # load state_dict
#     #     state_dict = torch.load(os.path.join(model_dir,"pytorch_model.bin"), map_location="cpu")
#     #     reg.load_state_dict(state_dict)
#     #     reg.eval().to(self.device)
#     #     # load y_keys
#     #     with open(os.path.join(os.path.dirname(model_dir),"target_map.json"), "r", encoding="utf-8") as f:
#     #         meta = json.load(f)
#     #     return tok, reg, meta["y_keys"]

#     def _load_reg(self, model_dir):
#     # 1. load tokenizer
#         tok = AutoTokenizer.from_pretrained(model_dir)
    
#         # 2. cari target y_keys
#         with open(os.path.join(os.path.dirname(model_dir), "target_map.json")) as f:
#             target_info = json.load(f)
#         y_keys = target_info["y_keys"]
    
#         # 3. bikin model skeleton dengan output dim yang sesuai
#         reg = BertRegressor(BASE_MODEL, out_dim=len(y_keys))
    
#         # 4. load weight
#         safetensor_path = os.path.join(model_dir, "model.safetensors")
    
#         import safetensors.torch as st
#         state_dict = st.load_file(safetensor_path, device="cpu")
    
#         reg.load_state_dict(state_dict)
    
#         # 5. eval mode + device
#         reg.eval().to(self.device)
    
#         return tok, reg, y_keys

#     @staticmethod
#     def _decode(i2l, idx: int):
#         # handle dict of {0:'labelA',1:'labelB',...}
#         if isinstance(i2l, dict):
#             return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
#         try:
#             return i2l[idx]
#         except:
#             return f"label_{idx}"

#     @torch.no_grad()
#     def _run_mc(self, tok, mdl, i2l, text, max_len=128):
#         enc = tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
#         logits = mdl(**enc).logits.squeeze(0)
#         probs = torch.softmax(logits, dim=0)
#         idx = int(torch.argmax(probs).item())
#         lab = self._decode(i2l, idx)
#         conf = float(probs[idx].item())
#         return lab, conf

#     @torch.no_grad()
#     def _run_reg(self, tok, mdl, ykeys, text, max_len=128):
#         enc = tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
#         out = mdl(**enc)
#         preds = out["logits"].squeeze(0).cpu().numpy().tolist()
#         return {ykeys[i]: float(preds[i]) for i in range(len(ykeys))}

#     def predict(self, text: str, max_len=128):
#         # 1. intent
#         intent_lab, intent_conf = self._run_mc(self.intent_tok, self.intent_m, self.intent_i2l, text, max_len)
#         # 2. scope
#         scope_lab, scope_conf   = self._run_mc(self.scope_tok, self.scope_m, self.scope_i2l, text, max_len)
#         # 3. time combo
#         timec_lab, timec_conf   = self._run_mc(self.timec_tok, self.timec_m, self.timec_i2l, text, max_len)
#         #    Pecah time combo biar downstream gampang
#         time_granularity, anchor_time_type, time_complexity, horizon_type = timec_lab.split("|")

#         # 4. horizon regression
#         horizon_pred            = self._run_reg(self.hor_tok, self.hor_m, self.hor_ykeys, text, max_len)
#         # 5. threshold stuff
#         thr_metric_lab, thr_metric_conf = self._run_mc(self.thmet_tok, self.thmet_m, self.thmet_i2l, text, max_len)
#         thr_oper_lab,  thr_oper_conf    = self._run_mc(self.thop_tok,  self.thop_m,  self.thop_i2l,  text, max_len)
#         thr_value_pred           = self._run_reg(self.thval_tok, self.thval_m, self.thval_y, text, max_len)

#         # 6. what-if flag
#         whatif_lab, whatif_conf = self._run_mc(self.wif_tok, self.wif_m, self.wif_i2l, text, max_len)
#         is_whatif = (whatif_lab == "IS_WHAT_IF")

#         # Route hints downstream → siapa yang ngerjain setelah ini?
#         route = []
#         # scope route
#         if scope_lab == "ALL":
#             route.append("aggregate_all_machines")
#         else:
#             route.append("machine_entity_extractor")

#         # intent route
#         if intent_lab == "FORECAST_VALUE":
#             route.append("forecast_numeric_pipeline")
#         elif intent_lab == "FORECAST_TREND":
#             route.append("forecast_trend_pipeline")
#         elif intent_lab == "THRESHOLD_TIME":
#             route.append("threshold_time_pipeline")
#         elif intent_lab == "WHAT_IF_FORECAST":
#             route.append("counterfactual_simulation_pipeline")

#         # time handling route
#         route.append(f"time_parse_{time_granularity.lower()}_{time_complexity.lower()}")

#         if is_whatif:
#             route.append("inject_counterfactual_state")

#         return {
#             "intent": {
#                 "label": intent_lab,
#                 "confidence": round(intent_conf,4)
#             },
#             "scope": {
#                 "label": scope_lab,
#                 "confidence": round(scope_conf,4)
#             },
#             "time": {
#                 "time_granularity": time_granularity,
#                 "anchor_time_type": anchor_time_type,
#                 "time_complexity": time_complexity,
#                 "horizon_type": horizon_type,
#                 "confidence": round(timec_conf,4)
#             },
#             "horizon_minutes": horizon_pred,  # {"y_start": ~3.0, "y_end": ~48.0} or {"y_val": ...}
#             "threshold": {
#                 "metric": {"label": thr_metric_lab, "confidence": round(thr_metric_conf,4)},
#                 "operator": {"label": thr_oper_lab, "confidence": round(thr_oper_conf,4)},
#                 "value_predicted": thr_value_pred  # {"y_val": ~80.0}
#             },
#             "what_if": {
#                 "flag": whatif_lab,
#                 "is_counterfactual": is_whatif,
#                 "confidence": round(whatif_conf,4)
#             },
#             "route_hints": route
#         }

# import re
# import math

# KNOWN_MACHINES = ["XP888A","PQ667","ZT901B","LM442","KJ3-A","ALPHA01","LINE-7"]

# def _extract_machines(user_text, scope_label):
#     """
#     Ambil nama mesin dari teks user.
#     - kalau scope ALL -> ["ALL"]
#     - kalau ada beberapa mesin disebut -> return list unik
#     - kalau tidak ada match tapi scope SINGLE/MULTI -> ["UNKNOWN"]
#     """
#     if scope_label == "ALL":
#         return ["ALL"]

#     found = []
#     for m in KNOWN_MACHINES:
#         # case-insensitive contain check, also handle things like "XP888A," or "XP888A?" dsb
#         pattern = r"\b" + re.escape(m) + r"\b"
#         if re.search(pattern, user_text, flags=re.IGNORECASE):
#             found.append(m)
#     found = list(dict.fromkeys(found))  # dedup, preserve order

#     if not found:
#         return ["UNKNOWN"]
#     return found

# def _snap_single_point(start_pred, end_pred):
#     """
#     Model kadang kasih 92 vs 101 menit padahal SINGLE_POINT.
#     Ambil rata-rata dan bulatkan.
#     """
#     avg_val = (float(start_pred) + float(end_pred)) / 2.0
#     avg_val = round(avg_val)
#     return avg_val, avg_val

# def _parse_explicit_horizon(user_text, time_info):
#     """
#     Coba deteksi horizon dari teks user.
#     Return (start_min, end_min, used_rule:bool) 
#     """
#     tcomp = time_info["time_complexity"]

#     txt = user_text.lower()

#     # pattern "<N> jam"
#     m_jam = re.search(r"(\d+)\s*(jam|jm)\b", txt)
#     # pattern "<N> menit" / mnt
#     m_menit = re.search(r"(\d+)\s*(menit|mnt)\b", txt)

#     # Helper to convert matched number
#     def _to_int(m):
#         try:
#             return int(m.group(1))
#         except:
#             return None

#     # CASE 1: menit-based horizon like "30 menit ke depan"
#     if m_menit:
#         n_min = _to_int(m_menit)
#         if n_min is not None:
#             # Check if wording is "ke depan", "sampai", "interval"
#             # -> treat as RANGE 0..n_min
#             if "ke depan" in txt or "kedepan" in txt or "sampai" in txt or "interval" in txt:
#                 start_min = 0
#                 end_min   = n_min
#             else:
#                 # fallback generic:
#                 if tcomp == "SINGLE_POINT":
#                     start_min = n_min
#                     end_min   = n_min
#                 else:
#                     start_min = 0
#                     end_min   = n_min
#             return start_min, end_min, True

#     # CASE 2: jam-based horizon like "2 jam lagi" / "2 jam ke depan"
#     if m_jam:
#         n_hour = _to_int(m_jam)
#         if n_hour is not None:
#             total_min = n_hour * 60

#             # kalau SINGLE_POINT → t+N jam
#             if tcomp == "SINGLE_POINT":
#                 start_min = total_min
#                 end_min   = total_min
#             else:
#                 # RANGE_REL → 0..N jam
#                 start_min = 0
#                 end_min   = total_min
#             return start_min, end_min, True

#     # CASE 3: "dalam <N> menit ke depan"
#     m_dlm = re.search(r"dalam\s+(\d+)\s*(menit|mnt)", txt)
#     if m_dlm:
#         n_min = int(m_dlm.group(1))
#         start_min = 0
#         end_min   = n_min
#         return start_min, end_min, True

#     # CASE 4: phrases "X menit depan", "X menit lagi"
#     m_simple_m = re.search(r"(\d+)\s*(menit|mnt)\s*(lagi|depan|kedepan)", txt)
#     if m_simple_m:
#         n_min = int(m_simple_m.group(1))
#         if tcomp == "SINGLE_POINT":
#             return n_min, n_min, True
#         else:
#             return 0, n_min, True

#     # CASE 5: phrases "X jam depan", "X jam kedepan"
#     m_simple_h = re.search(r"(\d+)\s*(jam|jm)\s*(lagi|depan|kedepan)", txt)
#     if m_simple_h:
#         n_h = int(m_simple_h.group(1))
#         tot = n_h * 60
#         if tcomp == "SINGLE_POINT":
#             return tot, tot, True
#         else:
#             return 0, tot, True

#     # kalau tidak ada match → return None
#     return None, None, False
# def extract_all_requested_minutes(user_text: str):
#     """
#     Cari semua 'N menit', 'N jam', 'N mnt', dst.
#     Return list menit (int).
#     """
#     # contoh pola: "48 menit", "24 menit", "12 menit", "2 jam", "1 jam", "30 mnt"
#     pattern = r'(\d+)\s*(menit|mnt|minute|min|jam|hour|h)'
#     hits = re.findall(pattern, user_text.lower())

#     mins = []
#     for num_str, unit in hits:
#         val = int(num_str)
#         if "jam" in unit or unit in ["hour","h"]:
#             mins.append(val * 60)
#         else:
#             mins.append(val)
#     # unique + sort
#     mins = sorted(list({m for m in mins}))
#     return mins
    
# def interpret_query(user_text: str, runner_output: dict):
#     """
#     user_text      : string original query user
#     runner_output  : dict hasil ForecastMultiRunner.predict()
#     return         : dict final NormalizedSpec
#     """

#     # 1. ambil komponen dari runner_output
#     intent_lab   = runner_output["intent"]["label"]
#     scope_lab    = runner_output["scope"]["label"]
#     time_info    = runner_output["time"]  # dict
#     horizon_pred = runner_output["horizon_minutes"]  # {'y_start':..., 'y_end':...}
#     whatif_info  = runner_output["what_if"]
#     thr_info     = runner_output["threshold"]

#     # 2. ekstrak mesin
#     machines = _extract_machines(user_text, scope_lab)

#     # 3. normalize horizon
#     #    3a. coba parse langsung dari teks
#     explicit_start, explicit_end, ok_rule = _parse_explicit_horizon(user_text, time_info)

#     if ok_rule:
#         final_start = explicit_start
#         final_end   = explicit_end
#         horizon_source = "text_rule"
#     else:
#         # fallback: pakai output model
#         y_s = float(horizon_pred.get("y_start", -1))
#         y_e = float(horizon_pred.get("y_end", -1))

#         # rapihin SINGLE_POINT
#         if time_info["time_complexity"] == "SINGLE_POINT":
#             y_s, y_e = _snap_single_point(y_s, y_e)

#         final_start = max(0, round(y_s)) if y_s >= 0 else -1
#         final_end   = max(final_start, round(y_e)) if y_e >= 0 else -1
#         horizon_source = "model_regressor"

#     # 4. threshold object
#     threshold_spec = None
#     if intent_lab == "THRESHOLD_TIME":
#         met = thr_info["metric"]["label"]
#         op  = thr_info["operator"]["label"]
#         val_pred = thr_info["value_predicted"].get("y_val", -1)

#         # map operator words → symbol
#         if op == "RISE_ABOVE":
#             op_symbol = ">"
#         elif op == "DROP_BELOW":
#             op_symbol = "<"
#         else:
#             op_symbol = None

#         # threshold_spec hanya valid kalau metric!=NONE dan op_symbol!=None
#         if met != "NONE" and op_symbol is not None and val_pred >= 0:
#             threshold_spec = {
#                 "metric": met,
#                 "operator": op_symbol,
#                 "value": round(float(val_pred), 2)
#             }

#     # 5. counterfactual flag
#     counterfactual_mode = whatif_info["is_counterfactual"]

#     # 6. action type (routing high level)
#     if intent_lab == "FORECAST_VALUE":
#         action_type = "forecast_numeric"
#     elif intent_lab == "FORECAST_TREND":
#         action_type = "forecast_trend"
#     elif intent_lab == "THRESHOLD_TIME":
#         action_type = "threshold_eta"
#     elif intent_lab == "WHAT_IF_FORECAST":
#         action_type = "counterfactual_forecast"
#     else:
#         action_type = "unknown"

#     # 7. build final normalized spec
#     spec = {
#         "action_type": action_type,
#         "intent": intent_lab,
#         "scope": scope_lab,
#         "machines": machines,  # list
#         "time": {
#             "granularity": time_info["time_granularity"],
#             "anchor_time_type": time_info["anchor_time_type"],
#             "time_complexity": time_info["time_complexity"],
#             "horizon_type": time_info["horizon_type"],
#             "window_minutes": {
#                 "start_min": final_start,
#                 "end_min": final_end,
#                 "source": horizon_source,
#             }
#         },
#         "threshold": threshold_spec,
#         "counterfactual": {
#             "enabled": counterfactual_mode,
#             # kamu bisa isi extra extraction di sini nanti (nilai override dsb)
#             "raw_query": user_text if counterfactual_mode else None
#         },
#         "route_hints": runner_output["route_hints"]
#     }
#     all_mins = extract_all_requested_minutes(user_text)
#     if len(all_mins) > 1:
#         # simpan list multi target
#         spec['time']['multi_targets_minutes'] = all_mins
#     else:
#         spec['time']['multi_targets_minutes'] = None

#     return spec
