# ============================================================
# pipeline_full.py
#
# End-to-end anomaly QA pipeline:
#   1. user query (bahasa natural)
#   2. intent classification (4-head model + smart rules)
#   3. time window resolution
#   4. anomaly scan (bandingkan data ts vs threshold)
#   5. payload siap buat LLM / UI dashboard
#
# Cocok buat:
# - smoke test manual di Colab
# - nanti di-wrap jadi API FastAPI/Flask
# ============================================================

from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import os
import re

import pandas as pd

from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# CONFIG
# -----------------------------
MODEL_ROOT   = BASE_DIR /"runs_anomaly_quad_relclass/runs_anomaly_quad_relclass"  # folder berisi 4 head final_model
TS_CSV_PATH  = BASE_DIR / "df_raw2.csv"        # hasil extractor df_raw2.csv
# TODO prod: load threshold dari Excel pakai loader threshold
DF_THRESHOLDS = pd.DataFrame({
    "machine_id": ["XP888A", "PQ667"],
    "max_value":  [10,       140],
    "min_value":  [0,        110],
})

# -----------------------------
# IMPORT MODUL INTERNAL
# -----------------------------


from query_interpret import smart_predict            # post-process intent
from time_parse import resolve_time_window, TimeWindow
import re
from anomaly_query_engine import run_anomaly_query   # orchestrator text→AnomalyResult
from other_ts_loader import load_timeseries_from_df_raw2
from payload_formatter import (
    anomaly_result_to_payload,
    pretty_print_payload,
)
# kata-kata yang mengindikasikan ada keterangan waktu di query
TIME_KEYWORDS = [
    "hari", "minggu", "bulan", "tahun",
    "kemarin", "tadi", "lalu", "terakhir",
    "shift", "pagi", "siang", "sore", "malam",
    "tanggal", "tgl", "jam",
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
]
TIME_REGEXES = [
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),      # 2025-01-15
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # 15/01/2025
    re.compile(r"\b\d{1,2}\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\b", re.I),
]

# ============================================================
# (1) Runtime model intent (Quad Head)
#    -> diambil dari versi kamu yang asli (:contentReference[oaicite:0]{index=0})
# ============================================================

class AnomalyQuadRunner:
    """
    Gabungan 4 head intent, rule-based:
      - scope            : ALL / NOT_ALL
      - time_granularity : HOUR / DAY / WEEK / MONTH / YEAR / SHIFT
      - time_complexity  : RANGE_CLEAR / RANGE_REL / WEEK_N / POINT / PERIOD / SHIFT
      - case             : RANGE_REL / RANGE_ABS / SINGLE_PERIOD / SUMMARY

    Signature & bentuk output dibuat kompatibel dengan versi lama
    yang pakai transformers, supaya modul lain tidak perlu diubah.
    Argumen scope_dir, gran_dir, cplx_dir, case_dir hanya dipertahankan
    agar signature sama, tapi TIDAK dipakai.
    """

    # ------- pola untuk CASE (dari train_case.jsonl) -------
    _MONTHS_ID = (
        "januari|februari|maret|april|mei|juni|juli|agustus|"
        "september|oktober|november|desember"
    )
    # Contoh: "Q1 2023"
    _RE_QUARTER = re.compile(r"\bq[1-4]\s*\d{4}\b", re.IGNORECASE)
    # Contoh: "Februari 10, 2024"
    _RE_FULLDATE = re.compile(
        rf"\b({_MONTHS_ID})\s+\d{{1,2}},\s*\d{{4}}\b",
        re.IGNORECASE,
    )

    def __init__(self, scope_dir: str = "", gran_dir: str = "", cplx_dir: str = "", case_dir: str = ""):
        # tidak ada model yang perlu diload
        pass

    # ------------------------------------------------------------------
    # HELPER: normalisasi lowercase
    # ------------------------------------------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        return text.lower().strip()

    # ------------------------------------------------------------------
    # 1) SCOPE HEAD: ALL vs NOT_ALL
    # ------------------------------------------------------------------
    def _predict_scope(self, text: str) -> (str, float):
        """
        RULE SCOPE:
        - Kalau ada kata 'semua' dan 'mesin' dalam kalimat
          (atau padanan 'seluruh mesin' / 'all machines'),
          -> scope = ALL
        - Selain itu -> NOT_ALL
        """
        t = self._norm(text)

        if (
            ("semua" in t and "mesin" in t)
            or "semua mesin" in t
            or "seluruh mesin" in t
            or "all machines" in t
        ):
            return "ALL", 0.96

        # sisanya dianggap subset / per-mesin
        return "NOT_ALL", 0.9

    # ------------------------------------------------------------------
    # 2) TIME GRANULARITY: HOUR / DAY / WEEK / MONTH / YEAR / SHIFT
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2) TIME GRANULARITY: HOUR / DAY / WEEK / MONTH / YEAR / SHIFT
    # ------------------------------------------------------------------
    def _predict_granularity(self, text: str) -> (str, float):
        """
        Granularity berdasarkan train_time_granularity.jsonl (disederhanakan):

        - HOUR  : konteks jam/menit, interval per jam, 15 menit, dsb.
        - DAY   : harian, per hari, daily, 'setiap hari', 'tiap hari'.
        - WEEK  : mingguan, per minggu, weekly, 'minggu', 'pekan'.
        - MONTH : bulanan, per bulan, monthly, bulan + nama bulan, quarter (Q1, Q2, ...).
        - YEAR  : tahunan, per tahun, yearly, 'tahun', 'tiap tahun'.
        - SHIFT : ada kata 'shift' (pagi/siang/malam/dll).

        Urutan prioritas:
          SHIFT > HOUR > WEEK > MONTH > YEAR > DAY > default DAY
        supaya kalimat tetap ditangkap natural.
        """
        t = self._norm(text)

        # --- SHIFT ---
        if "shift" in t:
            return "SHIFT", 0.95

        # --- HOUR (jam / menit / interval pendek) ---
        hour_keywords = [
            "pukul", "jam ", "jam:", "jam-", "jam ke-",
            "menit", "30 menit", "15 menit", "setengah jam",
            "per jam", "tiap jam", "setiap jam", "hourly"
        ]
        if any(k in t for k in hour_keywords):
            return "HOUR", 0.9

        # --- WEEK (mingguan / per minggu) ---
        week_keywords = [
            "per minggu", "tiap minggu", "setiap minggu",
            "mingguan", "weekly", "pekan", "per pekan"
        ]
        # kata "minggu" sendirian kita treat sebagai WEEK granularity
        if "minggu" in t or any(k in t for k in week_keywords):
            return "WEEK", 0.9

        # --- MONTH (bulanan / per bulan / nama bulan / quarter) ---
        month_words = [
            "januari", "februari", "maret", "april", "mei", "juni",
            "juli", "agustus", "september", "oktober", "november", "desember"
        ]
        month_keywords = [
            "per bulan", "tiap bulan", "setiap bulan",
            "bulanan", "monthly", "bulan ini", "bulan lalu"
        ]
        # quarter: Q1, Q2, Q3, Q4 kita map sebagai MONTH-level granularity
        has_quarter = re.search(r"\bq[1-4]\s*\d{4}\b", t) is not None

        if any(k in t for k in month_keywords) or has_quarter:
            return "MONTH", 0.9
        if any(m in t for m in month_words):
            # contoh "pada April 2024" atau "Agustus 2023"
            return "MONTH", 0.9

        # --- YEAR (tahunan / per tahun) ---
        year_keywords = [
            "per tahun", "tiap tahun", "setiap tahun",
            "tahunan", "yearly"
        ]
        if "tahun" in t or any(k in t for k in year_keywords):
            return "YEAR", 0.9

        # --- DAY (harian / per hari / default granularitas waktu pendek) ---
        day_keywords = [
            "per hari", "tiap hari", "setiap hari",
            "harian", "daily"
        ]
        if any(k in t for k in day_keywords):
            return "DAY", 0.9

        # Kalau ada kata 'hari', 'kemarin', 'tanggal', 'tgl' (tanpa petunjuk lain)
        if any(k in t for k in ["hari ", "hari-", "kemarin", "besok", "tanggal", "tgl "]):
            return "DAY", 0.85

        # --- fallback default ---
        # Kalau tidak jelas, asumsi defaultnya harian
        return "DAY", 0.6

    # ------------------------------------------------------------------
    # 3) TIME COMPLEXITY: RANGE_CLEAR / RANGE_REL / WEEK_N / POINT / PERIOD / SHIFT
    # ------------------------------------------------------------------
    def _predict_complexity(self, text: str) -> (str, float):
        """
        Diselaraskan dengan train_time_complexity.jsonl:

        LABEL:
          - SHIFT
          - RANGE_CLEAR   : rentang eksplisit (tanggal 5–20, rentang 1–12, dari X sampai Y)
          - RANGE_REL     : X hari/minggu/bulan terakhir, minggu lalu, selama N hari/bulan, dll.
          - PERIOD        : 1 bucket waktu (bulan lalu, bulan ini, April 2024, dst)
          - WEEK_N        : minggu/pekan ke-N
          - POINT         : titik waktu seperti "kemarin", "hari ini" tanpa rentang
        """
        t = self._norm(text)

        # 1) SHIFT: kalau ada kata 'shift'
        if "shift" in t:
            return "SHIFT", 0.95

        # 2) WEEK_N: minggu/pekan ke-N
        if re.search(r"(minggu|pekan)\s*(ke|-)\s*\d+", t):
            return "WEEK_N", 0.9

        # 3) RANGE_CLEAR: rentang eksplisit
        #    Contoh JSONL:
        #      - "rentang 1-12 Februari 2025"
        #      - "rentang 3-14 Agustus 2024"
        #      - "tanggal 5–20 Juli 2025"
        if "rentang" in t:
            return "RANGE_CLEAR", 0.92

        #    a) kata penghubung 'sampai', 'hingga', 's.d', 'sd'
        if any(k in t for k in [" sampai ", " s.d ", " s.d. ", " hingga ", " sd "]):
            return "RANGE_CLEAR", 0.9

        #    b) angka-angka dengan strip: "3-14", "5–20"
        if re.search(r"\b\d{1,2}\s*(–|-)\s*\d{1,2}\b", t):
            return "RANGE_CLEAR", 0.9

        # 4) RANGE_REL: frasa relatif "X hari/minggu/bulan terakhir", "minggu lalu", "selama ..."
        #    Contoh JSONL:
        #      - "60 hari terakhir", "3 minggu terakhir", "4 bulan terakhir"
        #      - "minggu lalu", "deteksi anomali ... minggu lalu"
        #      - "selama 60 hari terakhir", "selama 30 hari terakhir"
        is_relative_window = False

        if "selama" in t and any(
            unit in t for unit in ["hari", "minggu", "pekan", "bulan", "tahun"]
        ):
            is_relative_window = True

        if any(
            k in t
            for k in [
                "hari terakhir",
                "minggu terakhir",
                "bulan terakhir",
                "tahun terakhir",
                "minggu lalu",
                "pekan lalu",
                "dalam 7 hari",
                "dalam seminggu",
                "30 menit terakhir",
                "sebelumnya",
                "belakangan ini",
            ]
        ):
            is_relative_window = True

        if is_relative_window:
            return "RANGE_REL", 0.87

        # 5) POINT: satu titik waktu jelas tanpa rentang (kemarin, hari ini, dsb)
        #    (Ini lebih ke "sekali tembak" waktu pendek.)
        has_point_word = any(
            k in t
            for k in ["hari ini", "kemarin", "tadi malam", "pagi ini", "siang ini", "malam ini"]
        )
        has_range_marker = any(k in t for k in [" sampai ", "hingga", " s.d ", " sd ", "rentang", "selama"])
        if has_point_word and not has_range_marker:
            return "POINT", 0.8

        # 6) PERIOD: satu bucket waktu (bulan/tahun tertentu, bulan ini/lalu, secara bulanan)
        #    Contoh JSONL:
        #      - "bulan lalu"
        #      - "pada Agustus 2024", "pada April 2024", "pada Juli 2023"
        #      - "secara bulanan"
        month_words = [
            "januari", "februari", "maret", "april", "mei", "juni",
            "juli", "agustus", "september", "oktober", "november", "desember"
        ]

        # secara bulanan
        if "secara bulanan" in t:
            return "PERIOD", 0.9

        # bulan lalu / bulan ini
        if "bulan lalu" in t or "bulan ini" in t:
            return "PERIOD", 0.9

        # pola "pada <bulan> <tahun>" atau "<bulan> <tahun>"
        if any(f"pada {m}" in t for m in month_words):
            return "PERIOD", 0.9
        if any(m in t for m in month_words) and re.search(r"\b20\d{2}\b", t):
            return "PERIOD", 0.9

        # fallback: anggap satu periode (bucket)
        return "PERIOD", 0.7

    # ------------------------------------------------------------------
    # 4) CASE HEAD: RANGE_REL / RANGE_ABS / SINGLE_PERIOD / SUMMARY
    #     (disusun dari train_case.jsonl)
    # ------------------------------------------------------------------
    def _predict_case(self, text: str) -> (str, float):
        """
        Heuristik CASE berdasarkan train_case.jsonl:

        - SUMMARY:
            "bagian awal itu maksudnya tanggal berapa ya?"
        - RANGE_ABS:
            quarter (Q1 2023, Q3 2024, ...) atau tanggal absolut
            "Februari 10, 2024 jam 9 sampai 15"
        - SINGLE_PERIOD:
            "di bulan X ... dari awal/pertengahan/akhir bulan ..."
        - RANGE_REL:
            "selama 4 minggu terakhir", "pekan 3 sampai 5", "minggu 2 sampai 4",
            "tanggal 8–22 Oktober 2024", "bulan lalu", "tahun lalu", dll.
        """
        low = self._norm(text)

        # 1) SUMMARY – frasa khas
        if "bagian awal itu maksudnya" in low or "maksudnya tanggal berapa" in low:
            return "SUMMARY", 0.99

        # 2) RANGE_ABS – quarter dan tanggal penuh
        if self._RE_QUARTER.search(low):
            return "RANGE_ABS", 0.95
        if self._RE_FULLDATE.search(text):
            return "RANGE_ABS", 0.95

        # 3) SINGLE_PERIOD – di bulan X dari awal/pertengahan/akhir bulan
        if "di bulan" in low and any(
            phrase in low
            for phrase in ["awal bulan", "pertengahan bulan", "akhir bulan"]
        ):
            return "SINGLE_PERIOD", 0.95

        # 4) RANGE_REL – range relatif & range multi-bucket
        #    a) selama X minggu/hari/bulan terakhir
        if "selama" in low and any(
            w in low for w in ["minggu", "pekan", "hari", "bulan"]
        ):
            return "RANGE_REL", 0.9
        #    b) minggu/pekan terakhir
        if any(phrase in low for phrase in ["minggu terakhir", "pekan terakhir"]):
            return "RANGE_REL", 0.9
        #    c) pekan/minggu N sampai M
        if any(w in low for w in ["pekan", "minggu"]) and "sampai" in low:
            return "RANGE_REL", 0.9
        #    d) dari ... sampai ... tahun lalu/ini/terakhir
        if (
            "dari" in low
            and "sampai" in low
            and any(kw in low for kw in ["tahun lalu", "tahun ini", "terakhir"])
        ):
            return "RANGE_REL", 0.9
        #    e) tanggal rentang 10–20, 8-22, dst
        if re.search(r"\b\d{1,2}\s*(–|-)\s*\d{1,2}\b", low):
            return "RANGE_REL", 0.9

        # 5) fallback heuristik:
        #    - kalau ada 'sampai' tanpa pola lain → RANGE_REL
        if "sampai" in low or "hingga" in low or " s.d" in low or " sd " in low:
            return "RANGE_REL", 0.8

        #    - kalau konteksnya bulan/tahun tapi tanpa rentang eksplisit → SINGLE_PERIOD
        if "di bulan" in low or "bulan " in low:
            return "SINGLE_PERIOD", 0.8

        # terakhir, default ke RANGE_REL (label cukup umum)
        return "RANGE_REL", 0.6

    # ------------------------------------------------------------------
    # PREDICT: API KOMPATIBEL DENGAN VERSI TRANSFORMERS
    # ------------------------------------------------------------------
    def predict(self, text: str, max_len: int = 128) -> Dict[str, Any]:
        """
        Return struktur siap pakai downstream:
        {
          "scope": {"label": ..., "confidence": ...},
          "time_granularity": {"label": ..., "confidence": ...},
          "time_complexity": {"label": ..., "confidence": ...},
          "case": {"label": ..., "confidence": ...},
          "route_hints": [...],
        }
        """
        scope_label, scope_conf = self._predict_scope(text)
        gran_label, gran_conf = self._predict_granularity(text)
        cplx_label, cplx_conf = self._predict_complexity(text)
        case_label, case_conf = self._predict_case(text)

        # routing hints buat downstream pipeline (dipertahankan seperti versi lama)
        route: List[str] = []
        route.append(
            "aggregate_all_machines" if scope_label == "ALL" else "machine_entity_extractor"
        )
        if gran_label in {"HOUR", "DAY", "WEEK", "MONTH", "YEAR", "SHIFT"}:
            route.append(f"slot_{gran_label.lower()}_extractor")
        if cplx_label in {"RANGE_CLEAR", "RANGE_REL", "WEEK_N", "POINT", "PERIOD", "SHIFT"}:
            route.append(f"time_complexity_{cplx_label.lower()}")

        return {
            "scope": {"label": scope_label, "confidence": round(scope_conf, 4)},
            "time_granularity": {"label": gran_label, "confidence": round(gran_conf, 4)},
            "time_complexity": {"label": cplx_label, "confidence": round(cplx_conf, 4)},
            "case": {"label": case_label, "confidence": round(case_conf, 4)},
            "route_hints": route,
        }

# ============================================================
# (2) Bootstrap util
# ============================================================

def build_runner(model_root: str) -> AnomalyQuadRunner:
    """
    Load semua head classifier dari folder model_root.
    """
    return AnomalyQuadRunner(
        scope_dir=os.path.join(model_root, "scope", "final_model"),
        gran_dir=os.path.join(model_root, "time_granularity", "final_model"),
        cplx_dir=os.path.join(model_root, "time_complexity", "final_model"),
        case_dir=os.path.join(model_root, "case", "final_model"),
    )

def load_sensor_timeseries(ts_csv_path: str) -> pd.DataFrame:
    """
    Load dataframe sensor final (machine_id, ts[datetime64], value[float])
    dari df_raw2.csv yang udah dibersihin sebelumnya.
    """
    # asumsi timestamp historis = MM/DD/YYYY HH:MM
    # kalau ternyata DD/MM/YYYY ubah argumen assume_us_datetime=False
    return load_timeseries_from_df_raw2(
        ts_csv_path,
        assume_us_datetime=True,
        verbose=False,
    )


# ============================================================
# (3) Core: jalankan 1 query user end-to-end
# ============================================================

def run_query_once(
    query_text: str,
    df_ts: pd.DataFrame,
    df_thr: pd.DataFrame,
    runner: AnomalyQuadRunner,
):
    print("[bootstrap] CASE2 df_ts rows:", len(df_ts))
    print("[bootstrap] CASE2 df_thr rows:", len(df_thr))

    # 1) Interpret intent (scope, mesin, dsb.)
    pred = smart_predict(query_text, runner)

    # 2) Time window dari parser
    resolved_tw = resolve_time_window(query_text, pred)

    # 3) Cek apakah di query ada keterangan waktu eksplisit
    low_q = query_text.lower()
    has_time_word = any(kw in low_q for kw in TIME_KEYWORDS)
    has_time_regex = any(rgx.search(low_q) for rgx in TIME_REGEXES)
    has_time_expr = has_time_word or has_time_regex

    # 4) Data coverage (ts_min / ts_max) dari df_ts
    if df_ts.empty:
        ts_min = ts_max = None
    else:
        ts_min = df_ts["ts"].min()
        ts_max = df_ts["ts"].max()

    # 5) Fallback: kalau TIDAK ada keterangan waktu sama sekali di query,
    #    pakai seluruh rentang data yang tersedia sebagai window.
    if not has_time_expr and ts_min is not None and ts_max is not None:
        print("[TIME OVERRIDE] Query tanpa keterangan waktu → pakai seluruh data")
        resolved_tw = TimeWindow(
            start=ts_min,
            end=ts_max,
            granularity="DAY",
            kind="PERIOD",
            note="fallback: seluruh data yang tersedia",
        )

    # 6) Jalankan engine anomali dengan window yang sudah final
    result = run_anomaly_query(
        text=query_text,      # ganti dari query_text=...
        runner=runner,
        df_ts=df_ts,
        df_thr=df_thr,
        pred_override=pred,   # ganti dari pred=...
        resolve_time_window_fn=lambda _txt, _pred: resolved_tw,
    )



    # 7) Format payload untuk LLM/front-end
    payload = anomaly_result_to_payload(result, df_ts)
    return payload


# ============================================================
# (4) Demo batch (smoke test)
# ============================================================

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

def demo_batch():
    print("=== BOOTSTRAP MODEL & DATA ===")
    runner = build_runner(MODEL_ROOT)
    df_ts  = load_sensor_timeseries(TS_CSV_PATH)
    df_thr = DF_THRESHOLDS.copy()

    print("\n=== DEMO QUERY ===")
    for q in EXAMPLES:
        print("\n-------------------------------------------------")
        print("QUERY:", q)
        try:
            payload = run_query_once(
                query_text=q,
                runner=runner,
                df_ts=df_ts,
                df_thr=df_thr,
            )
            # tampilkan ringkas human-readable
            pretty_print_payload(payload)
        except Exception as e:
            print("ERROR:", e)

    print("\n[INFO]")
    print("Model root      :", MODEL_ROOT)
    print("Sensor rows     :", len(df_ts))
    print("Threshold rows  :", len(df_thr))


# ============================================================
# (5) Entry point
# ============================================================

if __name__ == "__main__":
    demo_batch()
