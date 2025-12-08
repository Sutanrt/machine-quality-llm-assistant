# =============================================================
# Forecast — Multi Head Trainer
# (intent • scope • timecombo • horizon • threshold • whatif)
# =============================================================
from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent

# ============ (0) CONFIG ============
DATA_DIR   = BASE_DIR/"next"   # ganti sesuai folder dataset hasil generator tadi
OUT_ROOT   = BASE_DIR/"runs_forecast_heads_v1"
BASE_MODEL = "indobenchmark/indobert-base-p1"     # atau "roberta-base" kalau mau english-ish

EPOCHS  = 20
LR      = 3e-5
BATCH   = 16
MAX_LEN = 128
SEED    = 42

import os, json, random, numpy as np, warnings
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_ROOT, exist_ok=True)

# -------------------------------------------------
# Helper: load jsonl (generic)
# -------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# -------------------------------------------------
# 1) MULTICLASS TRAINER
# -------------------------------------------------
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

def train_multiclass_head(model_name, train_jsonl, out_dir,
                          label_key="label", epochs=6, lr=3e-5,
                          batch=16, max_len=128, seed=42):
    """
    train_jsonl rows must have: {"text": "...", "label": "..."}
    We'll auto-split valid stratified 12%.
    """
    rows = load_jsonl(train_jsonl)
    train_rows, valid_rows = stratified_split(rows, label_key=label_key, test_size=0.12, seed=seed)

    # build huggingface DatasetDict
    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)
    ds = DatasetDict({"train": train_ds, "validation": valid_ds})

    # build label maps
    labels_sorted = sorted(list({ex[label_key] for ex in train_rows}))
    label2id = {l:i for i,l in enumerate(labels_sorted)}
    id2label = {i:l for l,i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(model_name)

    def _pp(batch):
        enc = tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = [label2id[y] for y in batch[label_key]]
        return enc

    enc = ds.map(_pp, batched=True, remove_columns=["text",label_key])
    cols = ["input_ids","attention_mask","labels"]
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

    # save final model
    final_dir = os.path.join(out_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    # classification report
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

    # label maps
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"id2label": id2label, "label2id": {v:k for k,v in id2label.items()}},
            f, ensure_ascii=False, indent=2
        )

    return final_dir, eval_res, id2label


# -------------------------------------------------
# 2) REGRESSION TRAINER (horizon_start_min / horizon_end_min, threshold_value)
# -------------------------------------------------

from torch import nn

class BertRegressor(nn.Module):
    """
    Simple regression head on top of base transformer.
    We'll reuse pretrained model's encoder via AutoModelForSequenceClassification's config trick:
    Actually easier: load AutoModelForSequenceClassification then replace classifier.
    BUT we'll just build from AutoModelForSequenceClassification and set num_labels = out_dim,
    and use MSE loss manually in Trainer via custom compute_loss.
    """
    def __init__(self, base_name, out_dim=1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(base_name)
        hidden = self.backbone.config.hidden_size
        self.reg_head = nn.Linear(hidden, out_dim)
        self.config = self.backbone.config
        self.out_dim = out_dim

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # use CLS token
        cls = out.last_hidden_state[:,0,:]
        preds = self.reg_head(cls)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels.float())
        return {"loss": loss, "logits": preds}


def make_regression_dataset(rows, y_keys):
    """
    rows: [{"text": "...", "y_start": 120, "y_end": 240}, ...]
    y_keys: list[str] of target names in each row -> we concat into vector
    returns DatasetDict with fields: text, targets(list[float])
    """
    X = []
    Y = []
    for r in rows:
        X.append(r["text"])
        Y.append([ float(r[k]) for k in y_keys ])
    return X, Y

def split_simple(rows, test_size=0.12, seed=42):
    # not stratified here (regression)
    idx = np.arange(len(rows))
    np.random.RandomState(seed).shuffle(idx)
    cut = int(len(rows)*(1-test_size))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    train_rows = [rows[i] for i in tr_idx]
    valid_rows = [rows[i] for i in va_idx]
    return train_rows, valid_rows

def train_regression_head(model_name, train_jsonl, out_dir,
                          y_keys, epochs=6, lr=3e-5, batch=16,
                          max_len=128, seed=42):
    """
    train_jsonl rows must have: {"text": "...", "y_start": ..., "y_end": ...}
    We'll auto split.
    y_keys = ["y_start","y_end"]  (multi-dim)  OR ["y_val"] (single)
    """
    rows = load_jsonl(train_jsonl)
    train_rows, valid_rows = split_simple(rows, test_size=0.12, seed=seed)

    Xtr, Ytr = make_regression_dataset(train_rows, y_keys)
    Xva, Yva = make_regression_dataset(valid_rows, y_keys)
    train_ds = Dataset.from_dict({"text": Xtr, "targets": Ytr})
    valid_ds = Dataset.from_dict({"text": Xva, "targets": Yva})
    ds = DatasetDict({"train": train_ds, "validation": valid_ds})

    tok = AutoTokenizer.from_pretrained(model_name)

    def _pp(batch):
        enc = tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = batch["targets"]  # we will convert to tensor later
        return enc

    enc = ds.map(_pp, batched=True, remove_columns=["text","targets"])
    # figure token_type_ids availability
    cols = ["input_ids","attention_mask","labels"]
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
        return {"mse": mse, "rmse": mse**0.5}

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

    # Simpan info dimensi target
    with open(os.path.join(out_dir, "target_map.json"), "w", encoding="utf-8") as f:
        json.dump({"y_keys": y_keys}, f, ensure_ascii=False, indent=2)

    return final_dir, eval_res, y_keys


# -------------------------------------------------
# (3) LATIH SEMUA HEAD
# -------------------------------------------------

paths = {
    "intent":                  (f"{DATA_DIR}/train_intent.jsonl",                "multiclass", {"label_key":"label"}),
    "scope":                   (f"{DATA_DIR}/train_scope.jsonl",                 "multiclass", {"label_key":"label"}),
    "timecombo":               (f"{DATA_DIR}/train_timecombo.jsonl",             "multiclass", {"label_key":"label"}),
    "whatif_flag":             (f"{DATA_DIR}/train_whatif_flag.jsonl",           "multiclass", {"label_key":"label"}),

    # threshold classification heads
    "threshold_metric":        (f"{DATA_DIR}/train_threshold_metric.jsonl",      "multiclass", {"label_key":"label"}),
    "threshold_operator":      (f"{DATA_DIR}/train_threshold_operator.jsonl",    "multiclass", {"label_key":"label"}),

    # regression heads
    "horizon":                 (f"{DATA_DIR}/train_horizon.jsonl",               "regress",   {"y_keys":["y_start","y_end"]}),
    "threshold_value":         (f"{DATA_DIR}/train_threshold_value.jsonl",       "regress",   {"y_keys":["y_val"]}),
}

models = {}

for head_name, (train_file, mode, extra) in paths.items():
    print(f"\n--- Training head: {head_name} ---")
    out_dir = os.path.join(OUT_ROOT, head_name)
    os.makedirs(out_dir, exist_ok=True)

    if mode == "multiclass":
        mdir, evres, i2l = train_multiclass_head(
            BASE_MODEL, train_file, out_dir,
            label_key=extra["label_key"],
            epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN, seed=SEED
        )
        models[head_name] = {"dir": mdir, "eval": evres, "i2l": i2l}

    elif mode == "regress":
        mdir, evres, ykeys = train_regression_head(
            BASE_MODEL, train_file, out_dir,
            y_keys=extra["y_keys"],
            epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN, seed=SEED
        )
        models[head_name] = {"dir": mdir, "eval": evres, "y_keys": ykeys}

print("\nTraining selesai semua head.")
print("Output root:", OUT_ROOT)


# -------------------------------------------------
# (4) RUNNER GABUNGAN BUAT INFERENCE
# -------------------------------------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

class ForecastMultiRunner:
    def __init__(self, root_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load multiclass heads
        self.intent_tok, self.intent_m, self.intent_i2l = self._load_mc(os.path.join(root_dir,"intent","final_model"))
        self.scope_tok, self.scope_m, self.scope_i2l = self._load_mc(os.path.join(root_dir,"scope","final_model"))
        self.timec_tok, self.timec_m, self.timec_i2l = self._load_mc(os.path.join(root_dir,"timecombo","final_model"))
        self.wif_tok, self.wif_m, self.wif_i2l       = self._load_mc(os.path.join(root_dir,"whatif_flag","final_model"))

        self.thmet_tok, self.thmet_m, self.thmet_i2l = self._load_mc(os.path.join(root_dir,"threshold_metric","final_model"))
        self.thop_tok,  self.thop_m,  self.thop_i2l  = self._load_mc(os.path.join(root_dir,"threshold_operator","final_model"))

        # load regression heads
        self.hor_tok, self.hor_m, self.hor_ykeys     = self._load_reg(os.path.join(root_dir,"horizon","final_model"))
        self.thval_tok, self.thval_m, self.thval_y   = self._load_reg(os.path.join(root_dir,"threshold_value","final_model"))

    def _load_mc(self, model_dir):
        cfg = AutoConfig.from_pretrained(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().to(self.device)
        i2l = cfg.id2label
        return tok, mdl, i2l

    def _load_reg(self, model_dir):
        # load tokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        # load custom BertRegressor
        # we re-init same arch used in training:
        reg = BertRegressor(BASE_MODEL, out_dim=len(json.load(open(os.path.join(os.path.dirname(model_dir),"target_map.json")))["y_keys"]))
        # load state_dict
        state_dict = torch.load(os.path.join(model_dir,"pytorch_model.bin"), map_location="cpu")
        reg.load_state_dict(state_dict)
        reg.eval().to(self.device)
        # load y_keys
        with open(os.path.join(os.path.dirname(model_dir),"target_map.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return tok, reg, meta["y_keys"]

    @staticmethod
    def _decode(i2l, idx: int):
        # handle dict of {0:'labelA',1:'labelB',...}
        if isinstance(i2l, dict):
            return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
        try:
            return i2l[idx]
        except:
            return f"label_{idx}"

    @torch.no_grad()
    def _run_mc(self, tok, mdl, i2l, text, max_len=128):
        enc = tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
        logits = mdl(**enc).logits.squeeze(0)
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.argmax(probs).item())
        lab = self._decode(i2l, idx)
        conf = float(probs[idx].item())
        return lab, conf

    @torch.no_grad()
    def _run_reg(self, tok, mdl, ykeys, text, max_len=128):
        enc = tok(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
        out = mdl(**enc)
        preds = out["logits"].squeeze(0).cpu().numpy().tolist()
        return {ykeys[i]: float(preds[i]) for i in range(len(ykeys))}

    def predict(self, text: str, max_len=128):
        # 1. intent
        intent_lab, intent_conf = self._run_mc(self.intent_tok, self.intent_m, self.intent_i2l, text, max_len)
        # 2. scope
        scope_lab, scope_conf   = self._run_mc(self.scope_tok, self.scope_m, self.scope_i2l, text, max_len)
        # 3. time combo
        timec_lab, timec_conf   = self._run_mc(self.timec_tok, self.timec_m, self.timec_i2l, text, max_len)
        #    Pecah time combo biar downstream gampang
        time_granularity, anchor_time_type, time_complexity, horizon_type = timec_lab.split("|")

        # 4. horizon regression
        horizon_pred            = self._run_reg(self.hor_tok, self.hor_m, self.hor_ykeys, text, max_len)
        # 5. threshold stuff
        thr_metric_lab, thr_metric_conf = self._run_mc(self.thmet_tok, self.thmet_m, self.thmet_i2l, text, max_len)
        thr_oper_lab,  thr_oper_conf    = self._run_mc(self.thop_tok,  self.thop_m,  self.thop_i2l,  text, max_len)
        thr_value_pred           = self._run_reg(self.thval_tok, self.thval_m, self.thval_y, text, max_len)

        # 6. what-if flag
        whatif_lab, whatif_conf = self._run_mc(self.wif_tok, self.wif_m, self.wif_i2l, text, max_len)
        is_whatif = (whatif_lab == "IS_WHAT_IF")

        # Route hints downstream → siapa yang ngerjain setelah ini?
        route = []
        # scope route
        if scope_lab == "ALL":
            route.append("aggregate_all_machines")
        else:
            route.append("machine_entity_extractor")

        # intent route
        if intent_lab == "FORECAST_VALUE":
            route.append("forecast_numeric_pipeline")
        elif intent_lab == "FORECAST_TREND":
            route.append("forecast_trend_pipeline")
        elif intent_lab == "THRESHOLD_TIME":
            route.append("threshold_time_pipeline")
        elif intent_lab == "WHAT_IF_FORECAST":
            route.append("counterfactual_simulation_pipeline")

        # time handling route
        route.append(f"time_parse_{time_granularity.lower()}_{time_complexity.lower()}")

        if is_whatif:
            route.append("inject_counterfactual_state")

        return {
            "intent": {
                "label": intent_lab,
                "confidence": round(intent_conf,4)
            },
            "scope": {
                "label": scope_lab,
                "confidence": round(scope_conf,4)
            },
            "time": {
                "time_granularity": time_granularity,
                "anchor_time_type": anchor_time_type,
                "time_complexity": time_complexity,
                "horizon_type": horizon_type,
                "confidence": round(timec_conf,4)
            },
            "horizon_minutes": horizon_pred,  # {"y_start": ~3.0, "y_end": ~48.0} or {"y_val": ...}
            "threshold": {
                "metric": {"label": thr_metric_lab, "confidence": round(thr_metric_conf,4)},
                "operator": {"label": thr_oper_lab, "confidence": round(thr_oper_conf,4)},
                "value_predicted": thr_value_pred  # {"y_val": ~80.0}
            },
            "what_if": {
                "flag": whatif_lab,
                "is_counterfactual": is_whatif,
                "confidence": round(whatif_conf,4)
            },
            "route_hints": route
        }

# ============ (5) DEMO INFERENCE ============
runner = ForecastMultiRunner(OUT_ROOT)

examples = [
    "nilai mesin XP888A 2 jam lagi berapa?",
    "coba prediksi semua mesin apakah akan ada anomali 2 jam kedepan?",
    "kapan XP888A bakal turun di bawah 70 derajat? kalau iya jam berapa pertama kalinya?",
    "andaikan data XP888A kemarin jam 10:00 gue ubah jadi 95, efeknya apa 2 jam lagi?",
    "ramalan performa semua mesin untuk 30 menit ke depan, interval 3 menit"
]

print("\n=== DEMO PREDIKSI ===")
for q in examples:
    print("\nQ:", q)
    print(runner.predict(q))
