# =============================================================
# intent_train.py
# Train 4-head intent classifier:
#   - scope
#   - time_granularity
#   - time_complexity
#   - case
# Output folder:
#   OUT_ROOT/<head>/final_model/
# =============================================================

from __future__ import annotations
import os, json, random, numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import torch


# -------------------------
# Config (bisa kamu override di Colab)
# -------------------------
DATA_DIR   = "/workspace/anomaly_comprehensive_ds_v2_relclass"
OUT_ROOT   = "./runs_anomaly_quad_relclass"
BASE_MODEL = "indobenchmark/indobert-base-p1"   # bisa ganti "roberta-base"

EPOCHS  = 30
LR      = 3e-5
BATCH   = 16
MAX_LEN = 128
SEED    = 42

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_ROOT, exist_ok=True)


# -------------------------
# Metrics utk Trainer
# -------------------------
def compute_metrics_mc(p):
    logits, y_true = p
    preds = logits.argmax(axis=1)
    return {
        "acc": accuracy_score(y_true, preds),
        "f1_macro": f1_score(y_true, preds, average="macro"),
    }


# -------------------------
# Core trainer 1 head
# -------------------------
def train_multiclass_from_jsonl(
    model_name: str,
    train_jsonl: str,
    valid_jsonl: Optional[str] = None,
    out_dir: str = "./out",
    epochs: int = 6,
    lr: float = 3e-5,
    batch: int = 16,
    max_len: int = 128,
    seed: int = 42,
):
    """
    Train satu classifier multi-class.
    - train_jsonl: path ke .jsonl berisi {"text": "...", "label": "..."}
    - valid_jsonl: optional .jsonl; kalau None atau file gak ada -> auto split 12% stratified

    Return:
        save_dir, eval_res, id2label, label2id
    """

    # ambil dataset
    if valid_jsonl and Path(valid_jsonl).exists():
        ds = load_dataset("json", data_files={
            "train": train_jsonl,
            "validation": valid_jsonl,
        })
    else:
        full = load_dataset("json", data_files={"train": train_jsonl})["train"]
        texts = full["text"]
        labels = full["label"]
        idx = np.arange(len(texts))

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.12,
            random_state=seed,
        )
        tr_idx, va_idx = next(sss.split(idx, labels))

        tr_rows = [{"text": texts[i], "label": labels[i]} for i in tr_idx]
        va_rows = [{"text": texts[i], "label": labels[i]} for i in va_idx]

        tmp_dir = Path(out_dir) / "_autosplit"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        trp = tmp_dir / "train.jsonl"
        vap = tmp_dir / "valid.jsonl"

        with open(trp, "w", encoding="utf-8") as f:
            for r in tr_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(vap, "w", encoding="utf-8") as f:
            for r in va_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ds = load_dataset(
            "json",
            data_files={"train": str(trp), "validation": str(vap)},
        )

    # buat kamus label
    labels_sorted = sorted(list({ex["label"] for ex in ds["train"]}))
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        enc["labels"] = [label2id[y] for y in batch["label"]]
        return enc

    enc = ds.map(_tokenize, batched=True, remove_columns=["text", "label"])

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in enc["train"].features:
        cols.append("token_type_ids")

    enc.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",     # <- perbaikan dari eval_strategy
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics_mc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    trainer.train()
    eval_res = trainer.evaluate()

    # simpan final_model/
    save_dir = os.path.join(out_dir, "final_model")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)

    # classification_report buat valid set
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

    # simpan label map
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id2label": id2label,
                "label2id": {v: k for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return save_dir, eval_res, id2label, {v: k for k, v in id2label.items()}


# -------------------------
# Training 4 head sekali jalan
# -------------------------
def train_all_heads(
    data_dir: str = DATA_DIR,
    out_root: str = OUT_ROOT,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch: int = BATCH,
    max_len: int = MAX_LEN,
    seed: int = SEED,
):
    """
    Latih 4 classifier:
      - scope
      - time_granularity
      - time_complexity
      - case
    """
    os.makedirs(out_root, exist_ok=True)

    paths = {
        "scope":            (f"{data_dir}/train_scope.jsonl",            None),
        "time_granularity": (f"{data_dir}/train_time_granularity.jsonl", None),
        "time_complexity":  (f"{data_dir}/train_time_complexity.jsonl",  None),
        "case":             (f"{data_dir}/train_case.jsonl",             None),
    }

    models = {}
    for head, (tr, va) in paths.items():
        print(f"\n--- Training head: {head} ---")
        mdir, ev, i2l, l2i = train_multiclass_from_jsonl(
            base_model,
            tr,
            va,
            out_dir=os.path.join(out_root, head),
            epochs=epochs,
            lr=lr,
            batch=batch,
            max_len=max_len,
            seed=seed,
        )
        print(f"Eval {head}:", ev)
        models[head] = {
            "dir": mdir,
            "eval": ev,
            "i2l": i2l,
            "l2i": l2i,
        }

    return models


if __name__ == "__main__":
    # optional: sanity check required dataset files
    req_files = [
        "train_scope.jsonl",
        "train_time_granularity.jsonl",
        "train_time_complexity.jsonl",
        "train_case.jsonl",
    ]
    for f in req_files:
        p = Path(DATA_DIR) / f
        print(f"{f}: {'OK' if p.exists() else 'MISSING'}  -> {p}")

    train_all_heads()
