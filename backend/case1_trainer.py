# trainer.py
# Script offline training tiga head (scope, time, case)
# Jalankan ini sekali untuk ngebangun model.

import os, json, csv, random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

# ------------------
# CONFIG
# ------------------
TRAIN_JSONL = "/workspace/train_trend_C_complex.jsonl"
LABEL_CSV   = "/workspace/labels_trend_C_index.enriched.csv"
OUT_ROOT    = "./runs_C_triple"
BASE_MODEL  = "indobenchmark/indobert-base-p1"

EPOCHS  = 6
LR      = 3e-5
BATCH   = 16
MAX_LEN = 128
SEED    = 42

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT_ROOT, exist_ok=True)

# ------------------
# DATA HELPERS
# ------------------

def load_jsonl(p: str) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def detect_delimiter_from_file(csv_path: str) -> str:
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
        except Exception:
            return ";"

def read_label_index_c(csv_path: str) -> Dict[str, Dict[str, str]]:
    delim = detect_delimiter_from_file(csv_path)
    idx = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            code = row.get("code") or row.get("kode")
            scope = row.get("scope")
            gran  = row.get("granularity") or row.get("granularitas")
            kasus = row.get("case") or row.get("kasus")
            if not (code and scope and gran and kasus):
                raise ValueError(f"Baris CSV tidak lengkap: {row}")
            idx[code] = {
                "scope": scope.upper(),
                "granularity": gran.upper(),
                "case": kasus.upper(),
            }
    return idx

def stratify_split(rows: List[Dict[str, Any]], test_size=0.12, seed=SEED):
    y = [r["label"] for r in rows]
    idxs = np.arange(len(rows))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(sss.split(idxs, y))
    tr = [rows[i] for i in tr_idx]
    va = [rows[i] for i in va_idx]
    return tr, va

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def build_three_datasets(train_jsonl: str, label_csv: str, out_dir: str,
                         valid_jsonl: str = "", auto_split=True) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    idx = read_label_index_c(label_csv)

    def convert(rows):
        scope_rows, time_rows, case_rows = [], [], []
        for r in rows:
            code = r["label"]
            if code not in idx:
                raise KeyError(f"Kode '{code}' tidak ditemukan di CSV index.")
            meta = idx[code]
            # scope -> ALL vs NOT_ALL
            scope_label = "ALL" if meta["scope"] == "ALL" else "NOT_ALL"
            time_label  = meta["granularity"]   # HOUR/DAY/WEEK/MONTH/YEAR/ALL
            case_label  = meta["case"]          # SINGLE_PERIOD / COMPARE / ...
            scope_rows.append({"text": r["text"], "label": scope_label})
            time_rows.append({"text": r["text"], "label": time_label})
            case_rows.append({"text": r["text"], "label": case_label})
        return scope_rows, time_rows, case_rows

    all_rows = load_jsonl(train_jsonl)
    if valid_jsonl and Path(valid_jsonl).exists():
        tr_rows = all_rows
        va_rows = load_jsonl(valid_jsonl)
    elif auto_split:
        tr_rows, va_rows = stratify_split(all_rows, test_size=0.12, seed=SEED)
    else:
        tr_rows, va_rows = all_rows, []

    tr_scope, tr_time, tr_case = convert(tr_rows)
    va_scope, va_time, va_case = convert(va_rows) if va_rows else ([], [], [])

    fp = {
        "train_scope": os.path.join(out_dir, "train_scope.jsonl"),
        "valid_scope": os.path.join(out_dir, "valid_scope.jsonl"),
        "train_time" : os.path.join(out_dir, "train_time.jsonl"),
        "valid_time" : os.path.join(out_dir, "valid_time.jsonl"),
        "train_case" : os.path.join(out_dir, "train_case.jsonl"),
        "valid_case" : os.path.join(out_dir, "valid_case.jsonl"),
    }

    save_jsonl(fp["train_scope"], tr_scope); save_jsonl(fp["valid_scope"], va_scope)
    save_jsonl(fp["train_time"],  tr_time);  save_jsonl(fp["valid_time"],  va_time)
    save_jsonl(fp["train_case"],  tr_case);  save_jsonl(fp["valid_case"],  va_case)

    return fp

# bangun dataset turunan
paths = build_three_datasets(
    TRAIN_JSONL,
    LABEL_CSV,
    out_dir=f"{OUT_ROOT}/data_C_triple",
    valid_jsonl="",
    auto_split=True,
)

# ------------------
# TRAINING
# ------------------

def compute_metrics_mc(p):
    logits, y = p
    preds = logits.argmax(axis=1)
    return {
        "acc": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
    }

def train_multiclass_from_jsonl(
    model_name, train_jsonl, valid_jsonl, out_dir,
    epochs=6, lr=3e-5, batch=16, max_len=128
):
    ds = load_dataset("json", data_files={"train": train_jsonl, "validation": valid_jsonl})
    labels = sorted(list({ex["label"] for ex in ds["train"]}))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for i,l in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(model_name)

    def _pp(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        enc["labels"] = [label2id[y] for y in batch["label"]]
        return enc

    enc = ds.map(_pp, batched=True, remove_columns=["text","label"])
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in enc["train"].features:
        cols.append("token_type_ids")
    enc.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch*2,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        metric_for_best_model="eval_f1_macro",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics_mc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    eval_res = trainer.evaluate()

    save_dir = os.path.join(out_dir, "final_model")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)

    pred_out = trainer.predict(enc["validation"])
    preds = np.argmax(pred_out.predictions, axis=1)
    y_true = pred_out.label_ids
    rep = classification_report(
        y_true,
        preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4,
    )

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"id2label": id2label, "label2id": label2id},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return save_dir, eval_res

# train tiap head
scope_dir, scope_eval = train_multiclass_from_jsonl(
    BASE_MODEL,
    paths["train_scope"], paths["valid_scope"],
    out_dir=os.path.join(OUT_ROOT, "scope"),
    epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN,
)

time_dir, time_eval = train_multiclass_from_jsonl(
    BASE_MODEL,
    paths["train_time"], paths["valid_time"],
    out_dir=os.path.join(OUT_ROOT, "time"),
    epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN,
)

case_dir, case_eval = train_multiclass_from_jsonl(
    BASE_MODEL,
    paths["train_case"], paths["valid_case"],
    out_dir=os.path.join(OUT_ROOT, "case"),
    epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN,
)

print("DONE TRAINING.")
print("scope model dir :", os.path.join(OUT_ROOT, "scope", "final_model"))
print("time model dir  :", os.path.join(OUT_ROOT, "time",  "final_model"))
print("case model dir  :", os.path.join(OUT_ROOT, "case",  "final_model"))


# # =============================================================
# # Pipeline C — Triple Head (Scope • Time • Case)
# # End-to-end: CONFIG -> DATA BUILD -> TRAIN (3 heads) -> RUNNER -> DEMO
# # =============================================================

# # ====== (0) CONFIG ======
# TRAIN_JSONL = "/workspace/train_trend_C_complex.jsonl"   # ganti jika perlu
# LABEL_CSV   = "/workspace/labels_trend_C_index.enriched.csv"      # ganti jika perlu
# OUT_ROOT    = "./runs_C_triple"                         # folder output
# BASE_MODEL  = "indobenchmark/indobert-base-p1"          # opsi lain: "roberta-base"

# EPOCHS  = 6
# LR      = 3e-5
# BATCH   = 16
# MAX_LEN = 128
# SEED    = 42

# # ====== (1) IMPORTS & SETUP ======
# import os, json, csv, random, itertools, math, sys, subprocess
# from pathlib import Path
# from typing import Dict, List, Any, Tuple
# import numpy as np
# random.seed(SEED); np.random.seed(SEED)

# print("=== Cek file input ===")
# print(" - JSONL:", Path(TRAIN_JSONL).exists(), TRAIN_JSONL)
# print(" - CSV  :", Path(LABEL_CSV).exists(), LABEL_CSV)

# # ====== (2) (Opsional) Install deps jika perlu ======
# # Catatan: Uncomment baris di bawah jika paket belum tersedia (atau jalankan manual).
# # !pip install -q transformers datasets accelerate scikit-learn

# # ====== (3) UTILITIES ======
# def load_jsonl(p: str) -> List[Dict[str, Any]]:
#     with open(p, "r", encoding="utf-8") as f:
#         return [json.loads(x) for x in f]

# def detect_delimiter_from_file(csv_path: str) -> str:
#     with open(csv_path, "r", encoding="utf-8") as f:
#         sample = f.read(2048)
#         f.seek(0)
#         try:
#             return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
#         except Exception:
#             return ";"

# def read_label_index_c(csv_path: str) -> Dict[str, Dict[str, str]]:
#     """
#     Kolom fleksibel (case-insensitive):
#       - code/kode
#       - scope
#       - granularity/granularitas
#       - case/kasus
#     Return: { code: {"scope":..., "granularity":..., "case":...} } (UPPER)
#     """
#     delim = detect_delimiter_from_file(csv_path)
#     idx = {}
#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter=delim)
#         for row in reader:
#             row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
#             code = row.get("code") or row.get("kode")
#             scope = row.get("scope")
#             gran  = row.get("granularity") or row.get("granularitas")
#             kasus = row.get("case") or row.get("kasus")
#             if not (code and scope and gran and kasus):
#                 raise ValueError(f"Baris CSV tidak lengkap: {row}")
#             idx[code] = {"scope": scope.upper(),
#                          "granularity": gran.upper(),
#                          "case": kasus.upper()}
#     return idx

# from sklearn.model_selection import StratifiedShuffleSplit
# def stratify_split(rows: List[Dict[str, Any]], test_size=0.12, seed=SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     y = [r["label"] for r in rows]
#     idxs = np.arange(len(rows))
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
#     tr_idx, va_idx = next(sss.split(idxs, y))
#     tr = [rows[i] for i in tr_idx]
#     va = [rows[i] for i in va_idx]
#     return tr, va

# def save_jsonl(path: str, rows: List[Dict[str, Any]]):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         for x in rows:
#             f.write(json.dumps(x, ensure_ascii=False) + "\n")

# def build_three_datasets(train_jsonl: str, label_csv: str, out_dir:str, valid_jsonl: str = "", auto_split=True) -> Dict[str, str]:
#     os.makedirs(out_dir, exist_ok=True)
#     idx = read_label_index_c(label_csv)

#     def convert(rows):
#         scope_rows, time_rows, case_rows = [], [], []
#         for r in rows:
#             code = r["label"]
#             if code not in idx:
#                 raise KeyError(f"Kode '{code}' tidak ditemukan di CSV index.")
#             meta = idx[code]
#             scope_label = "ALL" if meta["scope"] == "ALL" else "NOT_ALL"
#             time_label  = meta["granularity"]   # HOUR/DAY/WEEK/MONTH/YEAR/ALL
#             case_label  = meta["case"]          # mis. SINGLE_PERIOD, RANGE_ABS, RANGE_REL, ALL_TIME, COMPARE, SUMMARY
#             scope_rows.append({"text": r["text"], "label": scope_label})
#             time_rows.append({"text": r["text"], "label": time_label})
#             case_rows.append({"text": r["text"], "label": case_label})
#         return scope_rows, time_rows, case_rows

#     all_rows = load_jsonl(train_jsonl)
#     if valid_jsonl and Path(valid_jsonl).exists():
#         tr_rows = all_rows
#         va_rows = load_jsonl(valid_jsonl)
#     elif auto_split:
#         tr_rows, va_rows = stratify_split(all_rows, test_size=0.12, seed=SEED)
#     else:
#         tr_rows, va_rows = all_rows, []

#     tr_scope, tr_time, tr_case = convert(tr_rows)
#     va_scope, va_time, va_case = convert(va_rows) if va_rows else ([], [], [])

#     fp = {
#         "train_scope": os.path.join(out_dir, "train_scope.jsonl"),
#         "valid_scope": os.path.join(out_dir, "valid_scope.jsonl"),
#         "train_time" : os.path.join(out_dir, "train_time.jsonl"),
#         "valid_time" : os.path.join(out_dir, "valid_time.jsonl"),
#         "train_case" : os.path.join(out_dir, "train_case.jsonl"),
#         "valid_case" : os.path.join(out_dir, "valid_case.jsonl"),
#     }
#     save_jsonl(fp["train_scope"], tr_scope); save_jsonl(fp["valid_scope"], va_scope)
#     save_jsonl(fp["train_time"],  tr_time);  save_jsonl(fp["valid_time"],  va_time)
#     save_jsonl(fp["train_case"],  tr_case);  save_jsonl(fp["valid_case"],  va_case)
#     print(">> Tersimpan dataset turunan di:", out_dir)
#     return fp

# print("\n=== Preview data (3 baris JSONL & 6 baris CSV) ===")
# if Path(TRAIN_JSONL).exists():
#     with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
#         for i, line in zip(range(3), f):
#             print("[JSONL]", line.strip())
# if Path(LABEL_CSV).exists():
#     with open(LABEL_CSV, "r", encoding="utf-8") as f:
#         sample = f.read(2048); f.seek(0)
#         try:
#             dialect = csv.Sniffer().sniff(sample, delimiters=";,")
#         except Exception:
#             class _D: delimiter = ';'
#             dialect = _D()
#         reader = csv.reader(f, delimiter=dialect.delimiter)
#         for i, row in zip(range(6), reader):
#             print("[CSV]", row)

# # ====== (4) BUILD THREE DATASETS ======
# paths = build_three_datasets(
#     TRAIN_JSONL, LABEL_CSV,
#     out_dir=f"{OUT_ROOT}/data_C_triple",
#     valid_jsonl="", auto_split=True
# )
# print("paths:", paths)

# # ====== (5) TRAIN HELPERS & TRAINING 3 HEADS ======
# from datasets import load_dataset
# from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
#                           TrainingArguments, Trainer, EarlyStoppingCallback)
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# import torch

# def compute_metrics_mc(p):
#     logits, y = p
#     preds = logits.argmax(axis=1)
#     return {"acc": accuracy_score(y, preds), "f1_macro": f1_score(y, preds, average="macro")}

# def train_multiclass_from_jsonl(model_name, train_jsonl, valid_jsonl, out_dir,
#                                 epochs=6, lr=3e-5, batch=16, max_len=128):
#     ds = load_dataset("json", data_files={"train": train_jsonl, "validation": valid_jsonl})
#     labels = sorted(list({ex["label"] for ex in ds["train"]}))
#     label2id = {l:i for i,l in enumerate(labels)}
#     id2label = {i:l for i,l in enumerate(labels)}

#     tok = AutoTokenizer.from_pretrained(model_name)

#     def _pp(batch):
#         enc = tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
#         enc["labels"] = [label2id[y] for y in batch["label"]]
#         return enc

#     enc = ds.map(_pp, batched=True, remove_columns=["text","label"])
#     cols = ["input_ids","attention_mask","labels"]
#     if "token_type_ids" in enc["train"].features: cols.append("token_type_ids")
#     enc.set_format(type="torch", columns=cols)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
#     )

#     args = TrainingArguments(
#         output_dir=out_dir,
#         eval_strategy="epoch",
#         save_strategy="no",
#         load_best_model_at_end=False,
#         learning_rate=lr,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch,
#         per_device_eval_batch_size=batch*2,
#         weight_decay=0.01,
#         logging_steps=50,
#         report_to="none",
#         metric_for_best_model="eval_f1_macro",
#         seed=SEED,
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=enc["train"],
#         eval_dataset=enc["validation"],
#         tokenizer=tok,
#         compute_metrics=compute_metrics_mc,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
#     )

#     trainer.train()
#     eval_res = trainer.evaluate()

#     # save final
#     save_dir = os.path.join(out_dir, "final_model")
#     os.makedirs(save_dir, exist_ok=True)
#     trainer.save_model(save_dir)
#     tok.save_pretrained(save_dir)

#     # report
#     pred_out = trainer.predict(enc["validation"])
#     preds = np.argmax(pred_out.predictions, axis=1)
#     y_true = pred_out.label_ids
#     rep = classification_report(y_true, preds, target_names=[id2label[i] for i in range(len(id2label))], digits=4)
#     with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
#         f.write(rep)

#     with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
#         json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

#     return save_dir, eval_res, id2label, {v:k for k,v in id2label.items()}

# # ---- Train Scope
# scope_dir, scope_eval, scope_i2l, scope_l2i = train_multiclass_from_jsonl(
#     BASE_MODEL, paths["train_scope"], paths["valid_scope"],
#     out_dir=os.path.join(OUT_ROOT, "scope"),
#     epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN
# )
# print("\n== Eval Scope ==", scope_eval)

# # ---- Train Time
# time_dir, time_eval, time_i2l, time_l2i = train_multiclass_from_jsonl(
#     BASE_MODEL, paths["train_time"], paths["valid_time"],
#     out_dir=os.path.join(OUT_ROOT, "time"),
#     epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN
# )
# print("== Eval Time  ==", time_eval)

# # ---- Train Case
# case_dir, case_eval, case_i2l, case_l2i = train_multiclass_from_jsonl(
#     BASE_MODEL, paths["train_case"], paths["valid_case"],
#     out_dir=os.path.join(OUT_ROOT, "case"),
#     epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN
# )
# print("== Eval Case  ==", case_eval)

# # ====== (6) RUNNER (gabungan 3 head + route_hints) ======
# from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# class DualTripleRunner:
#     def __init__(self, scope_model_dir: str, time_model_dir: str, case_model_dir: str):
#         # Scope
#         self.scope_tok = AutoTokenizer.from_pretrained(scope_model_dir)
#         self.scope_cfg = AutoConfig.from_pretrained(scope_model_dir)
#         self.scope_i2l = self.scope_cfg.id2label
#         self.scope_m = AutoModelForSequenceClassification.from_pretrained(scope_model_dir).eval()

#         # Time
#         self.time_tok = AutoTokenizer.from_pretrained(time_model_dir)
#         self.time_cfg = AutoConfig.from_pretrained(time_model_dir)
#         self.time_i2l = self.time_cfg.id2label
#         self.time_m = AutoModelForSequenceClassification.from_pretrained(time_model_dir).eval()

#         # Case
#         self.case_tok = AutoTokenizer.from_pretrained(case_model_dir)
#         self.case_cfg = AutoConfig.from_pretrained(case_model_dir)
#         self.case_i2l = self.case_cfg.id2label
#         self.case_m = AutoModelForSequenceClassification.from_pretrained(case_model_dir).eval()

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.scope_m.to(self.device); self.time_m.to(self.device); self.case_m.to(self.device)

#     @staticmethod
#     def _decode(i2l, idx: int):
#         if isinstance(i2l, dict):
#             return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
#         try:
#             return i2l[idx]
#         except Exception:
#             return f"label_{idx}"

#     @torch.no_grad()
#     def predict(self, text: str, max_len=128):
#         # Scope
#         s = self.scope_tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
#         s_logits = self.scope_m(**s).logits.squeeze(0).cpu().numpy()
#         s_id = int(s_logits.argmax())
#         s_label = self._decode(self.scope_i2l, s_id)
#         s_conf = float(torch.softmax(torch.from_numpy(s_logits), dim=0)[s_id])

#         # Time
#         t = self.time_tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
#         t_logits = self.time_m(**t).logits.squeeze(0).cpu().numpy()
#         t_id = int(t_logits.argmax())
#         t_label = self._decode(self.time_i2l, t_id)
#         t_conf = float(torch.softmax(torch.from_numpy(t_logits), dim=0)[t_id])

#         # Case
#         c = self.case_tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
#         c_logits = self.case_m(**c).logits.squeeze(0).cpu().numpy()
#         c_id = int(c_logits.argmax())
#         c_label = self._decode(self.case_i2l, c_id)
#         c_conf = float(torch.softmax(torch.from_numpy(c_logits), dim=0)[c_id])

#         route = []
#         if c_label.startswith("RANGE"):
#             route.append("use_range_extractor")
#         if t_label in {"HOUR","DAY","WEEK","MONTH","YEAR"}:
#             route.append(f"slot_{t_label.lower()}_extractor")
#         if s_label == "ALL":
#             route.append("aggregate_all_machines")
#         else:
#             route.append("machine_entity_extractor")

#         return {
#             "scope": {"label": s_label, "confidence": round(s_conf, 4)},
#             "time":  {"label": t_label, "confidence": round(t_conf, 4)},
#             "case":  {"label": c_label, "confidence": round(c_conf, 4)},
#             "route_hints": route
#         }

# # ====== (7) DEMO INFERENCE ======
# scope_dir = os.path.join(OUT_ROOT, "scope", "final_model")
# time_dir  = os.path.join(OUT_ROOT, "time",  "final_model")
# case_dir  = os.path.join(OUT_ROOT, "case",  "final_model")
# runner = DualTripleRunner(scope_dir, time_dir, case_dir)

# examples = [
#     "Tolong tampilkan ringkasan semua mesin dari november sampai desember tahun lalu ini",
#     "Cek tren mesin XP888A hari ini",
#     "Apakah ada fluktuasi besar pada mesin XP888A dan PQ667 bulan ini?",
#     "Mohon tampilkan nilai tertinggi semua mesin tahun lalu",
#     "Tunjukkan tren mesin di bulan April tahun lalu dari awal bulan sampai pertengahan",
# ]
# print("\n=== DEMO PREDIKSI ===")
# for q in examples:
#     try:
#         print("\nQ:", q)
#         print(runner.predict(q))
#     except Exception as e:
#         print("Lewati (pastikan model terlatih & path benar):", e)

# print("\nSelesai. Model tersimpan di:", OUT_ROOT)
