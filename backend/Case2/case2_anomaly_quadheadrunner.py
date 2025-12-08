# =============================================================
# quadheadrunner.py / intent_runtime.py (RULE-BASED VERSION)
# Lightweight runtime inference for the 4-head intent "model".
# TIDAK memakai transformers atau torch; murni hardcode & regex.
# =============================================================

from __future__ import annotations
from typing import Dict, Any, List
import re


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


if __name__ == "__main__":
    # contoh sanity check lokal
    runner = AnomalyQuadRunner()

    examples = [
        "Tampilkan anomali mesin LMN77 dan QW12A bulan lalu",                         # PERIOD
        "Apakah ada anomali pada semua mesin 60 hari terakhir?",                     # RANGE_REL
        "Cek anomali mesin XR101 dan ZZ900 selama 60 hari terakhir",                 # RANGE_REL
        "Tolong tampilkan anomali semua mesin tanggal 5–20 Juli 2025",               # RANGE_CLEAR
        "Periksa mesin XP888A dan PQ667 rentang 1-12 Februari 2025 apakah ada anomali", # RANGE_CLEAR
        "Deteksi anomali semua mesin minggu lalu",                                   # RANGE_REL
        "Tampilkan anomali semua mesin shift kerja pagi",                            # SHIFT
        "Apakah mesin ZZ900 terlihat tidak normal bulan ini?",                       # PERIOD
        "Apakah ada anomali semua mesin pada shift malam?",                          # SHIFT
        "Apakah ada anomali pada semua mesin 3 hari terakhir?",                      # RANGE_REL
        "Apakah ada anomali mesin XR101 pada minggu ke-3 tahun ini?",                # WEEK_N (contoh)
        "Apakah ada anomali mesin XP888A kemarin malam?",                            # POINT
    ]
    for q in examples:
        print("\nQ:", q)
        print(runner.predict(q))
