# model_runner.py
# Improved, explainable hardcoded runner (NO transformers).
# Menggunakan pola dari train_time.jsonl untuk head "time".
#
# API:
#   from model_runner import runner
#   runner.predict(text)
#   runner.explain(text)

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

# ============================================================
# MACHINE WHITELIST (diambil dari dataset kamu)
# ============================================================
MACHINE_WHITELIST = {
    "PQ667", "XR101", "XP888A", "TRN-04", "LINE-07", "MCH-12"
}

# ============================================================
# KONFIGURASI
# ============================================================
DEFAULT_CONF_STRONG = 0.96
DEFAULT_CONF_MEDIUM = 0.90
DEFAULT_CONF_WEAK   = 0.75

PRIORITIZE_MACHINE_AS_NOT_ALL = True   # kalau ada ID mesin → NOT_ALL
ENABLE_YEAR = True
ENABLE_HOUR = True

# ============================================================
# REGEX & KATA KUNCI
# ============================================================
RE_MACHINE = re.compile(r"\b[A-Z]{1,}[A-Z0-9\-\_]*\d+[A-Z0-9\-\_]*\b")

RE_DATE       = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
RE_YEAR       = re.compile(r"\b(19|20)\d{2}\b")
RE_MONTH_WORD = re.compile(
    r"\b(JANUARI|FEBRUARI|MARET|APRIL|MEI|JUNI|JULI|AGUSTUS|SEPTEMBER|OKTOBER|NOVEMBER|DESEMBER|"
    r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
    re.I,
)
RE_QTR        = re.compile(r"\bQ[1-4]\b", re.I)
RE_RANGE_WORD = re.compile(r"\b(dari|sampai|hingga|to|until|between|–|-)\b", re.I)
RE_WEEK_NUM   = re.compile(r"\bMINGGU\s*\d+\b", re.I)

RELATIVE_MARKERS = ["LALU", "TERAKHIR", "KEMARIN", "TERAKHIRNYA", "LAST", "PAST", "AGO"]

HOUR_WORDS  = ["JAM", "PER JAM", "HOUR", "HOURS"]
DAY_WORDS   = ["HARI", "HARIAN", "DAY", "TANGGAL", "DATE"]
WEEK_WORDS  = ["MINGGU", "WEEK", "PEKAN"]
MONTH_WORDS = ["BULAN", "MONTH"]
YEAR_WORDS  = ["TAHUN", "YEAR"]

COMPARE_WORDS = ["BANDING", "VS", "VERSUS", "COMPARE", "PERBANDINGAN"]
SUMMARY_WORDS = ["RINGKAS", "RINGKASAN", "REKAP", "SUMMARY", "BAGIAN AWAL", "RINGKUM"]

# Label valid (buat clamp di akhir)
VALID_SCOPE = {"ALL", "NOT_ALL"}
VALID_TIME  = {"ALL", "DAY", "MONTH", "WEEK"}
if ENABLE_YEAR:
    VALID_TIME.add("YEAR")
if ENABLE_HOUR:
    VALID_TIME.add("HOUR")
VALID_CASE  = {"RANGE_ABS", "RANGE_REL", "SINGLE_PERIOD", "SUMMARY"}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def _txt(s: str) -> str:
    return (s or "").upper()

def extract_machines(text: str) -> List[str]:
    """Deteksi ID mesin dengan whitelist + heuristik."""
    t = _txt(text)
    candidates = RE_MACHINE.findall(t)

    # utamakan yang ada di whitelist
    filtered = [c for c in candidates if c in MACHINE_WHITELIST]

    # tambahkan kandidat lain yang kelihatannya ID mesin (bukan Q1..Q4)
    for c in candidates:
        if c in filtered:
            continue
        if ("-" in c or "_" in c) or (len(c) > 3 and any(ch.isdigit() for ch in c)):
            if re.match(r"Q[1-4]$", c):
                continue
            filtered.append(c)

    return filtered

def contains_any(text: str, keywords: List[str]) -> bool:
    T = _txt(text)
    return any(k.upper() in T for k in keywords)

def contains_month_word(text: str) -> bool:
    return bool(RE_MONTH_WORD.search(_txt(text)))

def contains_year(text: str) -> bool:
    if RE_YEAR.search(_txt(text)):
        return True
    return any(w in _txt(text) for w in YEAR_WORDS)

def contains_date(text: str) -> bool:
    # HANYA tanggal, quarter diproses terpisah (Q1/Q2/Q3/Q4 → ALL)
    return bool(RE_DATE.search(_txt(text)))

def _conf_for_signals(signals: List[str]) -> float:
    """Mapping sinyal → confidence."""
    score = 0.0
    for s in signals:
        if s in ("explicit_date", "explicit_range", "machine_id", "explicit_all", "relative_marker", "quarter"):
            score += 0.36
        elif s in ("month_word", "week_word", "day_word", "year_word"):
            score += 0.20
        elif s in ("compare_word", "summary_word"):
            score += 0.14
        else:
            score += 0.06

    if score >= 0.9:
        return DEFAULT_CONF_STRONG
    if score >= 0.7:
        return DEFAULT_CONF_MEDIUM
    return DEFAULT_CONF_WEAK

# ============================================================
# RUNNER
# ============================================================
class EnrichedHardcodedRunner:
    def __init__(self, prioritize_machine_as_not_all: bool = PRIORITIZE_MACHINE_AS_NOT_ALL):
        self.prioritize_machine_as_not_all = prioritize_machine_as_not_all

    # ---------- SCOPE ----------
    def _detect_scope(self, text: str) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        T = _txt(text)

        # "semua / seluruh / all" → ALL
        if any(x in T for x in ("SEMUA", "SELURUH", "ALL MACHINES", "ALL")):
            reasons.append("explicit_all")
            return "ALL", reasons

        # Kalau ketemu ID mesin → NOT_ALL
        machines = extract_machines(text)
        if machines:
            reasons.append(f"machine_id:{','.join(machines[:3])}")
            if self.prioritize_machine_as_not_all:
                reasons.append("machine_prioritized")
                return "NOT_ALL", reasons
            return "NOT_ALL", reasons

        # default
        reasons.append("default_all")
        return "ALL", reasons

    # ---------- TIME (pakai pola dari train_time.jsonl) ----------
    def _detect_time(self, text: str) -> Tuple[str, List[str]]:
        """
        Mapping ke label time:
        - ALL   : Q1/Q2/Q3/Q4, dan kalimat jenis "bagian awal itu maksudnya tanggal berapa ya?"
        - DAY   : ada tanggal spesifik / rentang tanggal / jam per hari
        - WEEK  : pekan X..Y, '3 minggu terakhir', dst.
        - MONTH : bulan tunggal atau rentang antar bulan
        - YEAR  : kalau ENABLE_YEAR dan ada tahun tapi tidak ada granularity lebih spesifik
        - HOUR  : kalau ENABLE_HOUR dan hanya bicara jam (tanpa tanggal)
        - ALL   : fallback ambigu
        """
        reasons: List[str] = []
        T = _txt(text)

        # 1) Pola special: quarter → ALL (lihat label ALL di train_time.jsonl)
        if RE_QTR.search(T):
            reasons.append("quarter")
            return "ALL", reasons

        # 2) Pola special: frasa 'bagian awal' (contoh: 'bulan April kemarin, bagian awal itu...') → ALL
        if "BAGIAN AWAL" in T:
            reasons.append("bagian_awal_ambiguous_all")
            return "ALL", reasons

        # 3) Explicit tanggal (12/03/2024, "tanggal 5–22 Mei 2024") → DAY
        if contains_date(T):
            reasons.append("explicit_date")
            return "DAY", reasons

        # 4) Tahun (jika diaktifkan) → YEAR (kecuali sudah ditangani date/quarter di atas)
        if ENABLE_YEAR and contains_year(T):
            reasons.append("year_word")
            return "YEAR", reasons

        # 5) Bulan (nama atau kata 'bulan', termasuk rentang 'Februari sampai September') → MONTH
        if contains_month_word(T) or contains_any(T, MONTH_WORDS):
            reasons.append("month_word")
            return "MONTH", reasons

        # 6) Minggu: pekan X..Y, '4 minggu terakhir', dst. → WEEK
        if contains_any(T, WEEK_WORDS) or RE_WEEK_NUM.search(T):
            reasons.append("week_word")
            return "WEEK", reasons

        # 7) Kata hari (hari ini, dll) → DAY
        if contains_any(T, DAY_WORDS):
            reasons.append("day_word")
            return "DAY", reasons

        # 8) Hanya jam (tanpa info tanggal) → HOUR (kalau diaktifkan)
        if ENABLE_HOUR and contains_any(T, HOUR_WORDS):
            reasons.append("hour_word")
            return "HOUR", reasons

        # 9) Fallback: ALL
        reasons.append("ambiguous_all")
        return "ALL", reasons

    # ---------- CASE ----------
    def _detect_case(self, text: str) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        T = _txt(text)

        # SUMMARY
        if contains_any(T, SUMMARY_WORDS) or "BAGIAN AWAL" in T:
            reasons.append("summary_word")
            return "SUMMARY", reasons

        # COMPARE
        if contains_any(T, COMPARE_WORDS):
            reasons.append("compare_word")
            # kalau ada tanggal/bulan eksplisit → kita anggap RANGE_ABS
            if RE_DATE.search(T) or contains_month_word(T):
                reasons.append("abs_date_present")
                return "RANGE_ABS", reasons
            return "RANGE_REL", reasons

        # RANGE dengan kata 'dari...sampai...' atau ada tanggal
        if RE_RANGE_WORD.search(T) or RE_DATE.search(T):
            reasons.append("range_indicator_or_date")
            if any(r in T for r in RELATIVE_MARKERS):
                reasons.append("relative_marker")
                return "RANGE_REL", reasons
            return "RANGE_ABS", reasons

        # '4 minggu terakhir', '6 bulan terakhir', dsb.
        if re.search(r"\b\d+\s*(MINGGU|WEEK|BULAN|MONTH|TAHUN|YEAR)\b", T):
            reasons.append("numeric_relative_period")
            return "RANGE_REL", reasons

        # Ada kata periode (bulan/minggu/hari/Q*) → SINGLE_PERIOD
        if any(w in T for w in ("BULAN", "MINGGU", "HARI", "Q1", "Q2", "Q3", "Q4")):
            reasons.append("period_word")
            return "SINGLE_PERIOD", reasons

        # fallback
        reasons.append("fallback_summary")
        return "SUMMARY", reasons

    # ---------- PREDICT (API utama) ----------
    def predict(self, text: str, max_len: int = 128) -> Dict[str, Any]:
        scope_label, scope_reasons = self._detect_scope(text)
        time_label,  time_reasons  = self._detect_time(text)
        case_label,  case_reasons  = self._detect_case(text)

        # kumpulkan sinyal untuk confidence
        scope_signals: List[str] = []
        if "explicit_all" in scope_reasons: scope_signals.append("explicit_all")
        if any(r.startswith("machine_id") for r in scope_reasons): scope_signals.append("machine_id")
        if "machine_prioritized" in scope_reasons: scope_signals.append("machine_prioritized")
        if "default_all" in scope_reasons: scope_signals.append("default")

        time_signals: List[str] = []
        for r in time_reasons:
            if r in ("explicit_date", "quarter"): time_signals.append("explicit_date")
            elif r.endswith("_word"): time_signals.append(r)
            elif r == "ambiguous_all": time_signals.append("ambiguous")

        case_signals: List[str] = []
        for r in case_reasons:
            if r in ("range_indicator_or_date", "numeric_relative_period"): case_signals.append("explicit_range")
            if r == "relative_marker": case_signals.append("relative_marker")
            if r == "compare_word": case_signals.append("compare_word")
            if r in ("summary_word", "fallback_summary"): case_signals.append("summary_word")

        scope_conf = _conf_for_signals(scope_signals)
        time_conf  = _conf_for_signals(time_signals)
        case_conf  = _conf_for_signals(case_signals)

        # clamp label supaya aman
        if scope_label not in VALID_SCOPE:
            scope_label = "ALL"
        if time_label not in VALID_TIME:
            time_label = "ALL"
        if case_label not in VALID_CASE:
            case_label = "SUMMARY"

        # route_hints untuk pipeline downstream
        route: List[str] = []
        if case_label.startswith("RANGE"):
            route.append("use_range_extractor")
        if time_label != "ALL":
            route.append(f"slot_{time_label.lower()}_extractor")
        if scope_label == "ALL":
            route.append("aggregate_all_machines")
        else:
            route.append("machine_entity_extractor")

        return {
            "scope": {"label": scope_label, "confidence": round(scope_conf, 3)},
            "time":  {"label": time_label,  "confidence": round(time_conf, 3)},
            "case":  {"label": case_label,  "confidence": round(case_conf, 3)},
            "route_hints": route,
        }

    # ---------- EXPLAIN (opsional, buat debugging) ----------
    def explain(self, text: str) -> Dict[str, Any]:
        scope_label, scope_reasons = self._detect_scope(text)
        time_label,  time_reasons  = self._detect_time(text)
        case_label,  case_reasons  = self._detect_case(text)
        machines = extract_machines(text)

        scope_signals = ["machine_id"] if any(r.startswith("machine_id") for r in scope_reasons) else []
        scope_conf    = _conf_for_signals(scope_signals)
        time_conf     = _conf_for_signals(time_reasons)
        case_conf     = _conf_for_signals(case_reasons)

        return {
            "scope": {
                "label": scope_label,
                "reasons": scope_reasons,
                "machines": machines,
                "confidence": round(scope_conf, 3),
            },
            "time": {
                "label": time_label,
                "reasons": time_reasons,
                "confidence": round(time_conf, 3),
            },
            "case": {
                "label": case_label,
                "reasons": case_reasons,
                "confidence": round(case_conf, 3),
            },
            "machines": machines,
        }

# ============================================================
# GLOBAL RUNNER INSTANCE
# ============================================================
runner = EnrichedHardcodedRunner()

# Quick test ketika file dijalankan langsung
if __name__ == "__main__":
    import json
    samples = [
        "Tunjukkan tren mesin XP888A dari bulan Mei sampai Desember tahun lalu.",
        "Tolong tampilkan tren semua mesin di bulan April kemarin, bagian awal itu maksudnya tanggal berapa ya?",
        "Analisis tren mesin XR101 pada Q1 2023.",
        "Tampilkan mesin XP888A pada Maret 12, 2024 jam 9 sampai 15.",
        "Ringkas tren XP888A dan XR101 selama 4 minggu terakhir.",
    ]
    for s in samples:
        print("-" * 60)
        print("QUERY:", s)
        print("PREDICT:", json.dumps(runner.predict(s), ensure_ascii=False))
        print("EXPLAIN:", json.dumps(runner.explain(s), ensure_ascii=False, indent=2))


# # model_runner.py
# # Loader inference untuk triple-head classifier.
# # File ini aman diimport dari main_pipeline.py

# import os
# import torch
# from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# class DualTripleRunner:
#     """
#     Wrapper 3-head:
#       - scope: ALL vs NOT_ALL
#       - time:  HOUR/DAY/WEEK/MONTH/YEAR/ALL
#       - case:  SINGLE_PERIOD / COMPARE / RANGE_... / SUMMARY / ...
#     Output sesuai format yang dipakai interpret().
#     """

#     def __init__(self, scope_model_dir: str, time_model_dir: str, case_model_dir: str, device: str = None):
#         # pilih device
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = torch.device(device)

#         # Scope head
#         self.scope_tok = AutoTokenizer.from_pretrained(scope_model_dir)
#         self.scope_cfg = AutoConfig.from_pretrained(scope_model_dir)
#         self.scope_i2l = self.scope_cfg.id2label
#         self.scope_m = AutoModelForSequenceClassification.from_pretrained(scope_model_dir).to(self.device).eval()

#         # Time head
#         self.time_tok = AutoTokenizer.from_pretrained(time_model_dir)
#         self.time_cfg = AutoConfig.from_pretrained(time_model_dir)
#         self.time_i2l = self.time_cfg.id2label
#         self.time_m = AutoModelForSequenceClassification.from_pretrained(time_model_dir).to(self.device).eval()

#         # Case head
#         self.case_tok = AutoTokenizer.from_pretrained(case_model_dir)
#         self.case_cfg = AutoConfig.from_pretrained(case_model_dir)
#         self.case_i2l = self.case_cfg.id2label
#         self.case_m = AutoModelForSequenceClassification.from_pretrained(case_model_dir).to(self.device).eval()

#     @staticmethod
#     def _decode(i2l, idx: int):
#         # i2l bisa dict {0:"ALL",1:"NOT_ALL"} / mapping string-key dsb
#         if isinstance(i2l, dict):
#             return i2l.get(idx) or i2l.get(str(idx)) or f"label_{idx}"
#         try:
#             return i2l[idx]
#         except Exception:
#             return f"label_{idx}"

#     @torch.no_grad()
#     def predict(self, text: str, max_len: int = 128):
#         # ---- Scope head ----
#         s = self.scope_tok(
#             text,
#             truncation=True,
#             padding=True,
#             max_length=max_len,
#             return_tensors="pt",
#         ).to(self.device)
#         s_logits = self.scope_m(**s).logits.squeeze(0)
#         s_probs = torch.softmax(s_logits, dim=0)
#         s_id = int(torch.argmax(s_probs).item())
#         s_label = self._decode(self.scope_i2l, s_id)
#         s_conf = float(s_probs[s_id].item())

#         # ---- Time head ----
#         t = self.time_tok(
#             text,
#             truncation=True,
#             padding=True,
#             max_length=max_len,
#             return_tensors="pt",
#         ).to(self.device)
#         t_logits = self.time_m(**t).logits.squeeze(0)
#         t_probs = torch.softmax(t_logits, dim=0)
#         t_id = int(torch.argmax(t_probs).item())
#         t_label = self._decode(self.time_i2l, t_id)
#         t_conf = float(t_probs[t_id].item())

#         # ---- Case head ----
#         c = self.case_tok(
#             text,
#             truncation=True,
#             padding=True,
#             max_length=max_len,
#             return_tensors="pt",
#         ).to(self.device)
#         c_logits = self.case_m(**c).logits.squeeze(0)
#         c_probs = torch.softmax(c_logits, dim=0)
#         c_id = int(torch.argmax(c_probs).item())
#         c_label = self._decode(self.case_i2l, c_id)
#         c_conf = float(c_probs[c_id].item())

#         # route_hints (opsional, useful downstream)
#         route = []
#         if c_label.startswith("RANGE"):
#             route.append("use_range_extractor")
#         if t_label in {"HOUR", "DAY", "WEEK", "MONTH", "YEAR"}:
#             route.append(f"slot_{t_label.lower()}_extractor")
#         if s_label == "ALL":
#             route.append("aggregate_all_machines")
#         else:
#             route.append("machine_entity_extractor")

#         return {
#             "scope": {"label": s_label, "confidence": round(s_conf, 4)},
#             "time":  {"label": t_label, "confidence": round(t_conf, 4)},
#             "case":  {"label": c_label, "confidence": round(c_conf, 4)},
#             "route_hints": route,
#         }


# # ==================================================
# # GLOBAL RUNNER INSTANCE
# # ==================================================
# #
# # Ini diasumsikan model sudah dilatih pakai trainer.py
# # dan disimpan di:
# #   runs_C_triple/scope/final_model
# #   runs_C_triple/time/final_model
# #   runs_C_triple/case/final_model
# #
# # Kamu bisa ubah path ini kalau lokasi model beda.

# _DEFAULT_SCOPE_DIR = "/workspace/runs_C_triple/scope/final_model"
# _DEFAULT_TIME_DIR  = "/workspace/runs_C_triple/time/final_model"
# _DEFAULT_CASE_DIR  = "/workspace/runs_C_triple/case/final_model"

# runner = DualTripleRunner(
#     scope_model_dir=_DEFAULT_SCOPE_DIR,
#     time_model_dir=_DEFAULT_TIME_DIR,
#     case_model_dir=_DEFAULT_CASE_DIR,
# )
