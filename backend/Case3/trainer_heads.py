# trainer_heads.py
import os, json, random, numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error
)
from sklearn.model_selection import StratifiedShuffleSplit

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from .config import (
    DATA_DIR, OUT_ROOT, BASE_MODEL,
    EPOCHS, LR, BATCH, MAX_LEN, SEED
)

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_ROOT, exist_ok=True)

# ---------- generic util ----------
def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# ---------- multiclass training ----------
def compute_metrics_mc(p):
    logits, y = p
    preds = logits.argmax(axis=1)
    return {
        "acc": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro")
    }

def stratified_split(rows, label_key="label", test_size=0.12, seed=42):
    labels = [r[label_key] for r in rows]
    idx = np.arange(len(rows))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(sss.split(idx, labels))
    train_rows = [rows[i] for i in tr_idx]
    valid_rows = [rows[i] for i in va_idx]
    return train_rows, valid_rows

def train_multiclass_head(
    model_name: str,
    train_jsonl: str,
    out_dir: str,
    label_key: str = "label",
    epochs: int = 6,
    lr: float = 3e-5,
    batch: int = 16,
    max_len: int = 128,
    seed: int = 42
):
    """
    train_jsonl rows must have:
      {"text": "...", "<label_key>": "..."}
    We'll auto-split valid stratified 12%.
    """
    rows = load_jsonl(train_jsonl)
    train_rows, valid_rows = stratified_split(
        rows,
        label_key=label_key,
        test_size=0.12,
        seed=seed
    )

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)
    ds = DatasetDict({"train": train_ds, "validation": valid_ds})

    labels_sorted = sorted(list({ex[label_key] for ex in train_rows}))
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(model_name)

    def _pp(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        enc["labels"] = [label2id[y] for y in batch[label_key]]
        return enc

    enc = ds.map(_pp, batched=True, remove_columns=["text", label_key])
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in enc["train"].features:
        cols.append("token_type_ids")
    enc.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch*2,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        metric_for_best_model="eval_f1_macro",
        save_total_limit=2,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics_mc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    trainer.train()
    eval_res = trainer.evaluate()

    final_dir = os.path.join(out_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    pred_out = trainer.predict(enc["validation"])
    preds = np.argmax(pred_out.predictions, axis=1)
    y_true = pred_out.label_ids
    rep = classification_report(
        y_true,
        preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4
    )
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id2label": id2label,
                "label2id": {v: k for k, v in id2label.items()}
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return final_dir, eval_res, id2label

# ---------- regression training ----------
class BertRegressor(nn.Module):
    """
    Regression head di atas backbone transformer.
    Output dim = len(y_keys).
    """
    def __init__(self, base_name: str, out_dim: int = 1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(base_name)
        hidden = self.backbone.config.hidden_size
        self.reg_head = nn.Linear(hidden, out_dim)
        self.config = self.backbone.config
        self.out_dim = out_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0, :]
        preds = self.reg_head(cls)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels.float())
        return {"loss": loss, "logits": preds}

def _make_regression_dataset(rows, y_keys):
    X, Y = [], []
    for r in rows:
        X.append(r["text"])
        Y.append([float(r[k]) for k in y_keys])
    return X, Y

def _split_simple(rows, test_size=0.12, seed=42):
    idx = np.arange(len(rows))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    cut = int(len(rows) * (1 - test_size))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    train_rows = [rows[i] for i in tr_idx]
    valid_rows = [rows[i] for i in va_idx]
    return train_rows, valid_rows

def train_regression_head(
    model_name: str,
    train_jsonl: str,
    out_dir: str,
    y_keys,
    epochs=6,
    lr=3e-5,
    batch=16,
    max_len=128,
    seed=42
):
    """
    train_jsonl rows must have fields for y_keys:
      ex: {"text": "...", "y_start": 30, "y_end": 120}
    """
    rows = load_jsonl(train_jsonl)
    train_rows, valid_rows = _split_simple(rows, test_size=0.12, seed=seed)

    Xtr, Ytr = _make_regression_dataset(train_rows, y_keys)
    Xva, Yva = _make_regression_dataset(valid_rows, y_keys)

    train_ds = Dataset.from_dict({"text": Xtr, "targets": Ytr})
    valid_ds = Dataset.from_dict({"text": Xva, "targets": Yva})
    ds = DatasetDict({"train": train_ds, "validation": valid_ds})

    tok = AutoTokenizer.from_pretrained(model_name)

    def _pp(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        enc["labels"] = batch["targets"]
        return enc

    enc = ds.map(_pp, batched=True, remove_columns=["text", "targets"])
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in enc["train"].features:
        cols.append("token_type_ids")
    enc.set_format(type="torch", columns=cols)

    model = BertRegressor(model_name, out_dim=len(y_keys))

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch*2,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
        seed=seed,
    )

    def compute_metrics_reg(p):
        preds = p.predictions
        y = p.label_ids
        mse = mean_squared_error(y, preds)
        return {
            "mse": mse,
            "rmse": mse ** 0.5
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics_reg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    trainer.train()
    eval_res = trainer.evaluate()

    final_dir = os.path.join(out_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    with open(os.path.join(out_dir, "target_map.json"), "w", encoding="utf-8") as f:
        json.dump({"y_keys": y_keys}, f, ensure_ascii=False, indent=2)

    return final_dir, eval_res, y_keys

def train_all_heads():
    """
    Latih semua head (intent, scope, timecombo, whatif_flag,
    threshold_metric, threshold_operator,
    horizon, threshold_value)
    sesuai mapping paths.
    """
    paths = {
        "intent":            (f"{DATA_DIR}/train_intent.jsonl",            "multiclass", {"label_key": "label"}),
        "scope":             (f"{DATA_DIR}/train_scope.jsonl",             "multiclass", {"label_key": "label"}),
        "timecombo":         (f"{DATA_DIR}/train_timecombo.jsonl",         "multiclass", {"label_key": "label"}),
        "whatif_flag":       (f"{DATA_DIR}/train_whatif_flag.jsonl",       "multiclass", {"label_key": "label"}),
        "threshold_metric":  (f"{DATA_DIR}/train_threshold_metric.jsonl",  "multiclass", {"label_key": "label"}),
        "threshold_operator":(f"{DATA_DIR}/train_threshold_operator.jsonl","multiclass", {"label_key": "label"}),
        "horizon":           (f"{DATA_DIR}/train_horizon.jsonl",           "regress",   {"y_keys":["y_start","y_end"]}),
        "threshold_value":   (f"{DATA_DIR}/train_threshold_value.jsonl",   "regress",   {"y_keys":["y_val"]}),
    }

    models = {}
    for head_name, (train_file, mode, extra) in paths.items():
        print(f"\n--- Training head: {head_name} ---")
        out_dir = os.path.join(OUT_ROOT, head_name)
        os.makedirs(out_dir, exist_ok=True)

        if mode == "multiclass":
            mdir, evres, i2l = train_multiclass_head(
                BASE_MODEL,
                train_file,
                out_dir,
                label_key=extra["label_key"],
                epochs=EPOCHS,
                lr=LR,
                batch=BATCH,
                max_len=MAX_LEN,
                seed=SEED
            )
            models[head_name] = {"dir": mdir, "eval": evres, "i2l": i2l}

        elif mode == "regress":
            mdir, evres, ykeys = train_regression_head(
                BASE_MODEL,
                train_file,
                out_dir,
                y_keys=extra["y_keys"],
                epochs=EPOCHS,
                lr=LR,
                batch=BATCH,
                max_len=MAX_LEN,
                seed=SEED
            )
            models[head_name] = {"dir": mdir, "eval": evres, "y_keys": ykeys}

    return models
