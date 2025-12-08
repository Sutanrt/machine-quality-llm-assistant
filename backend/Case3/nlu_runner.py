# nlu_runner.py
from __future__ import annotations
import re
from typing import Dict, Any, Tuple


class ForecastMultiRunner:
    """
    Versi lightweight (rule-based) dari NLU runner untuk case forecast.

    Perbedaan utama:
    - TIDAK lagi load model HuggingFace / transformers.
    - Semua head (intent, scope, timecombo, horizon, threshold, what-if)
      diputuskan dengan aturan keyword (hardcode) yang disusun supaya
      selaras dengan pola di dataset train_*.jsonl.

    API tetap sama:
      runner = ForecastMultiRunner(root_dir)
      out = runner.predict("teks query ...")

    Output `predict()`:
    {
      "intent": {...},
      "scope": {...},
      "time": {...},
      "horizon_minutes": {"y_start": float, "y_end": float},
      "threshold": {
        "metric": {...},
        "operator": {...},
        "value_predicted": {"y_val": float},
      },
      "what_if": {...},
      "route_hints": [...]
    }
    """

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #
    def __init__(self, root_dir: str, device: str | None = None):
        """
        root_dir dan device dipertahankan hanya untuk kompatibilitas,
        tapi sekarang tidak dipakai karena semuanya pure rule-based.
        """
        self.root_dir = root_dir
        self.device = device or "cpu"

    # ------------------------------------------------------------------ #
    # HELPER UMUM
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm(text: str) -> str:
        return (text or "").strip().lower()

    # ------------------------------------------------------------------ #
    # 1) INTENT HEAD (FORECAST_VALUE / FORECAST_TREND / THRESHOLD_TIME / WHAT_IF_FORECAST)
    # ------------------------------------------------------------------ #
    def _rule_intent(self, text: str) -> Tuple[str, float]:
        t = self._norm(text)

        # ---- WHAT-IF ----
        if any(
            kw in t
            for kw in [
                "what if",
                "andaikan",
                "andaikan kalau",
                "seandainya",
                "misalnya kalau",
                "kalau misalnya",
                "kalau gue ubah",
                "kalau saya ubah",
                "kalau datanya saya ganti",
            ]
        ):
            return "WHAT_IF_FORECAST", 0.95

        # ---- THRESHOLD_TIME (kapan melewati ambang / di atas / di bawah) ----
        if any(kw in t for kw in ["kapan", "jam berapa", "menit berapa"]) and any(
            kw in t
            for kw in [
                "di bawah",
                "dibawah",
                "di atas",
                "diatas",
                "lebih dari",
                "kurang dari",
                "melewati",
                "melampaui",
                ">",
                "<",
            ]
        ):
            return "THRESHOLD_TIME", 0.93

        # ---- FORECAST_TREND (tren / pola / grafik ke depan) ----
        if any(
            kw in t
            for kw in [
                "tren",
                "trend",
                "pola",
                "pergerakan",
                "grafik",
                "perubahan",
                "naik turun",
            ]
        ) and any(kw in t for kw in ["ke depan", "kedepan", "ke depannya", "ke depan."]):
            return "FORECAST_TREND", 0.9

        # ---- default: FORECAST_VALUE ----
        # Frasa umum: "berapa nanti", "nilai 2 jam lagi", "prediksi"
        if any(
            kw in t
            for kw in [
                "berapa",
                "nilai",
                "angka",
                "prediksi",
                "ramalan",
                "forecast",
                "proyeksi",
            ]
        ) or any(kw in t for kw in ["menit lagi", "jam lagi", "hari lagi", "ke depan"]):
            return "FORECAST_VALUE", 0.9

        # fallback aman
        return "FORECAST_VALUE", 0.6

    # ------------------------------------------------------------------ #
    # 2) SCOPE HEAD (ALL / SINGLE / MULTI)
    # ------------------------------------------------------------------ #
    def _rule_scope(self, text: str) -> Tuple[str, float]:
        t = self._norm(text)

        # Kalau eksplisit "semua" → ALL (permintaanmu: "kalau ada kata semua ya semua mesin")
        if "semua mesin" in t or "semua line" in t or "semua equipment" in t:
            return "ALL", 0.98
        if re.search(r"\bsemua\b", t):
            return "ALL", 0.95

        # deteksi entity mesin sederhana: token uppercase + angka (XP888A, PQ667, dll.)
        machine_like = re.findall(r"\b[A-Z]{2}\d{2,}[A-Z0-9]*\b", text)
        uniq = list(dict.fromkeys(machine_like))  # dedup & preserve order

        if len(uniq) == 0:
            # tidak ada mesin eksplisit → default ALL (biar pipeline agregasi)
            return "ALL", 0.7
        if len(uniq) == 1:
            return "SINGLE", 0.9
        return "MULTI", 0.92

    # ------------------------------------------------------------------ #
    # 3) TIME + HORIZON (timecombo + horizon_minutes)
    # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
    # 3) TIME + HORIZON (timecombo + horizon_minutes)
    # ------------------------------------------------------------------
    _NUM_UNIT_RE = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(menit|min|mnt|minute|minutes|jam|jm|hour|hours|hari|day|days)\b",
        re.I,
    )

    @staticmethod
    def _unit_to_minutes(value: float, unit: str) -> float:
        u = unit.lower()
        if u.startswith(("menit", "min", "mnt", "minute")):
            return value
        if u.startswith(("jam", "jm", "hour")):
            return value * 60.0
        if u.startswith(("hari", "day")):
            return value * 24.0 * 60.0
        return value

    def _rule_time_and_horizon(
        self, text: str
    ) -> Tuple[str, str, str, str, float, Dict[str, float]]:
        """
        Menghasilkan:
        - time_granularity: MINUTE | HOUR | DAY
        - anchor_time_type : NOW_START | ABSOLUTE_START
        - time_complexity  : SINGLE_POINT | RANGE_FUTURE | RANGE_REL | RANGE_ABSOLUTE
        - horizon_type     : POINT | RANGE
        - conf             : float
        - horizon_pred     : {"y_start": ..., "y_end": ...} (dalam menit)

        Diselaraskan dengan pola di train_horizon.jsonl:
        - "selama 48 menit ke depan" / "dalam 48 menit ke depan" -> [0, 48]
        - "2 jam ke depan ada anomali?" -> [0, 120]
        - "1 jam lagi berapa?" -> [60, 60]
        - explicit window (08:00-12:00, 1-7 Jan, shift ini, dll) -> [-1, -1]
        """

        t = self._norm(text)

        # ---------- 0. Deteksi query anomali / gejala ----------
        is_anomaly_query = any(
            kw in t
            for kw in [
                "anomali",
                "gejala abnormal",
                "abnormal",
                "ngaco",
                "nggak normal",
                "tidak normal",
                "perilaku aneh",
                "perilaku nggak normal",
                "overheat",
                "keluar batas",
                "keluar dari batas normal",
            ]
        )

        # ---------- 1. Deteksi explicit window --> [-1, -1] ----------
        # contoh di dataset: "besok 08:00-12:00", "1–7 Jan 2026", "shift ini (1-7 Jan 2026)"
        has_time_range = bool(
            re.search(r"\b\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\b", text)
        )
        has_day_month_range = bool(
            re.search(
                r"\b\d{1,2}\s*[-–]\s*\d{1,2}\s+"
                r"(januari|februari|maret|april|mei|juni|juli|agustus|"
                r"september|oktober|november|desember)\b",
                t,
            )
        )
        has_week_range = bool(
            re.search(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\s+minggu\b", t)
        )
        has_shift_window = "shift ini" in t or "window " in t

        if has_time_range or has_day_month_range or has_week_range or has_shift_window:
            # Granularity & complexity masih kita isi wajar, tapi horizon menit diserahkan ke time parser
            gran = "HOUR" if has_time_range else "DAY"
            return (
                gran,
                "ABSOLUTE_START",
                "RANGE_ABSOLUTE",
                "RANGE",
                0.95,
                {"y_start": -1.0, "y_end": -1.0},
            )

        # ---------- 2. Cari angka + unit (menit/jam/hari) ----------
        matches = list(self._NUM_UNIT_RE.finditer(text))

        # Kalau tidak ada angka-unit --> horizon default pendek-menengah
        if not matches:
            # default: 0–60 menit ke depan
            return (
                "MINUTE",
                "NOW_START",
                "RANGE_FUTURE",
                "RANGE",
                0.4,
                {"y_start": 0.0, "y_end": 60.0},
            )

        # helper pilih granularity
        def pick_gran(unit: str) -> str:
            u = unit.lower()
            if u.startswith(("menit", "min", "mnt", "minute")):
                return "MINUTE"
            if u.startswith(("jam", "jm", "hour")):
                return "HOUR"
            if u.startswith(("hari", "day")):
                return "DAY"
            return "MINUTE"

        # deteksi range relatif ke belakang: "30 menit terakhir", "6 jam terakhir"
        is_past_range = any(
            kw in t for kw in ["terakhir", "kebelakang", "ke belakang", "sebelumnya"]
        )

        # ---------- 3. Satu angka + unit ----------
        if len(matches) == 1:
            m = matches[0]
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2)
            minutes = self._unit_to_minutes(val, unit)
            gran = pick_gran(unit)

            # 3.a) Range ke belakang: "30 menit terakhir" -> [-30, 0]
            if is_past_range:
                return (
                    gran,
                    "NOW_START",
                    "RANGE_REL",
                    "RANGE",
                    0.9,
                    {"y_start": -minutes, "y_end": 0.0},
                )

            # 3.b) "selama / dalam / untuk X menit|jam ke depan" -> [0, X]
            if any(kw in t for kw in ["selama", "dalam", "untuk"]) and any(
                kw in t
                for kw in [
                    "ke depan",
                    "kedepan",
                    "menit depan",
                    "jam depan",
                    "jam kedepan",
                    "jam ke depan",
                ]
            ):
                return (
                    gran,
                    "NOW_START",
                    "RANGE_FUTURE",
                    "RANGE",
                    0.95,
                    {"y_start": 0.0, "y_end": minutes},
                )

            # 3.c) Query anomali: "2 jam ke depan ada anomali?" -> [0, 120]
            if is_anomaly_query and any(
                kw in t
                for kw in [
                    "ke depan",
                    "kedepan",
                    "menit depan",
                    "jam depan",
                    "jam kedepan",
                    "jam ke depan",
                ]
            ):
                return (
                    gran,
                    "NOW_START",
                    "RANGE_FUTURE",
                    "RANGE",
                    0.95,
                    {"y_start": 0.0, "y_end": minutes},
                )

            # 3.d) Default: forecast nilai titik "1 jam lagi berapa?" -> [60, 60]
            return (
                gran,
                "NOW_START",
                "SINGLE_POINT",
                "POINT",
                0.95,
                {"y_start": minutes, "y_end": minutes},
            )

        # ---------- 4. >= 2 angka + unit (range atau multi horizon) ----------
        # Contoh di dataset:
        # - "3, 12, 24, 30 menit"   -> ~[3, 30] (di label sering 6–30)
        # - "30 sampai 120 menit"   -> [30, 120]
        # - "dalam 48 menit ke depan" (jarang multi, tapi jaga-jaga)
        vals_minutes = []
        for m in matches:
            v = float(m.group(1).replace(",", "."))
            u = m.group(2)
            vals_minutes.append(self._unit_to_minutes(v, u))

        m_min = min(vals_minutes)
        m_max = max(vals_minutes)
        gran = pick_gran(matches[-1].group(2))

        # 4.a) Query dengan interval 3 menit -> target 6–max (mendekati label 6–30)
        if any(
            kw in t
            for kw in [
                "interval per 3 menit",
                "tiap 3 menit interval",
                "tiap 3 menit",
                "setiap 3 menit",
            ]
        ):
            start = 6.0 if m_min <= 6.0 <= m_max else m_min
            return (
                "MINUTE",
                "NOW_START",
                "RANGE_FUTURE",
                "RANGE",
                0.95,
                {"y_start": float(start), "y_end": float(m_max)},
            )

        # 4.b) Range ke belakang: "30-120 menit terakhir" -> treat sebagai RANGE_REL (kasus jarang)
        if is_past_range:
            return (
                gran,
                "NOW_START",
                "RANGE_REL",
                "RANGE",
                0.9,
                {"y_start": -float(m_max), "y_end": 0.0},
            )

        # 4.c) Default multi-value: future window min-max
        return (
            gran,
            "NOW_START",
            "RANGE_FUTURE",
            "RANGE",
            0.95,
            {"y_start": float(m_min), "y_end": float(m_max)},
        )


    # ------------------------------------------------------------------ #
    # 4) THRESHOLD METRIC / OPERATOR / VALUE
    # ------------------------------------------------------------------ #
    def _rule_threshold_metric(self, text: str) -> Tuple[str, float]:
        t = self._norm(text)

        if any(kw in t for kw in ["suhu", "temperatur", "temperature", "derajat"]):
            return "TEMP", 0.9

        if any(kw in t for kw in ["pressure", "tekanan", "bar", "psi"]):
            return "PRESSURE", 0.85

        if any(kw in t for kw in ["arus", "ampere", "amp", "current"]):
            return "CURRENT", 0.85

        if any(kw in t for kw in ["vibrasi", "getaran", "vibration"]):
            return "VIBRATION", 0.85

        # fallback generik
        return "VALUE", 0.6

    def _rule_threshold_operator(self, text: str) -> Tuple[str, float]:
        t = self._norm(text)

        # mapping label mengikuti style umum: GT / LT / GE / LE
        if any(
            kw in t
            for kw in [
                "di atas",
                "diatas",
                "lebih dari",
                "lebih besar dari",
                ">= ",
                ">=",
                " > ",
                "melampaui",
                "melewati",
            ]
        ):
            return "GT", 0.9

        if any(
            kw in t
            for kw in [
                "di bawah",
                "dibawah",
                "kurang dari",
                "lebih kecil dari",
                "<= ",
                "<=",
                " < ",
            ]
        ):
            return "LT", 0.9

        # fallback: kalau tidak eksplisit tapi ada angka threshold → GT
        if re.search(r"\b\d+(?:[.,]\d+)?\b", t):
            return "GT", 0.5

        return "NONE", 0.3

    _THRESH_VALUE_RE = re.compile(
        r"(?:di\s+(?:bawah|atas)|lebih\s+dari|kurang\s+dari|>=|<=|>|<)\s*"
        r"(\d+(?:[.,]\d+)?)",
        re.I,
    )

    def _rule_threshold_value(self, text: str) -> Dict[str, float]:
        """
        Ambil angka threshold (misal 'di bawah 70 derajat', 'lebih dari 80').
        Return: {"y_val": float}
        """
        m = self._THRESH_VALUE_RE.search(text)
        if m:
            val = float(m.group(1).replace(",", "."))
            return {"y_val": val}

        # fallback kalau tidak ketemu: nilai default "netral"
        return {"y_val": 80.0}

    # ------------------------------------------------------------------ #
    # 5) WHAT-IF FLAG
    # ------------------------------------------------------------------ #
    def _rule_whatif_flag(self, text: str) -> Tuple[str, float]:
        t = self._norm(text)
        if any(
            kw in t
            for kw in [
                "what if",
                "andaikan",
                "seandainya",
                "kalau misalnya",
                "misalnya kalau",
                "kalau gue ubah",
                "kalau saya ubah",
                "jika saya ubah",
            ]
        ):
            return "IS_WHAT_IF", 0.96
        return "NOT_WHAT_IF", 0.99

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #
    def predict(self, text: str, max_len: int = 128) -> Dict[str, Any]:
        """
        Jalankan semua "head" dengan aturan rule-based dan kembalikan struktur
        yang sama dengan versi transformer-based sebelumnya.
        """

        # 1. intent
        intent_lab, intent_conf = self._rule_intent(text)

        # 2. scope
        scope_lab, scope_conf = self._rule_scope(text)

        # 3. time + horizon_minutes
        (
            time_granularity,
            anchor_time_type,
            time_complexity,
            horizon_type,
            time_conf,
            horizon_pred,
        ) = self._rule_time_and_horizon(text)

        # 4. threshold metric + operator + value
        thr_metric_lab, thr_metric_conf = self._rule_threshold_metric(text)
        thr_oper_lab, thr_oper_conf = self._rule_threshold_operator(text)
        thr_value_pred = self._rule_threshold_value(text)

        # 5. what-if
        whatif_lab, whatif_conf = self._rule_whatif_flag(text)
        is_whatif = whatif_lab == "IS_WHAT_IF"

        # 6. route_hints (tetap sama seperti sebelumnya)
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

        # time parse route
        route.append(f"time_parse_{time_granularity.lower()}_{time_complexity.lower()}")

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
                "confidence": round(time_conf, 4),
            },
            # dulu horizon_minute = hasil regresi; sekarang diganti hasil rule (menit ke depan)
            "horizon_minutes": horizon_pred,  # ex: {"y_start": 3.0, "y_end": 48.0}
            "threshold": {
                "metric": {
                    "label": thr_metric_lab,
                    "confidence": round(thr_metric_conf, 4),
                },
                "operator": {
                    "label": thr_oper_lab,
                    "confidence": round(thr_oper_conf, 4),
                },
                "value_predicted": thr_value_pred,  # {"y_val": ~80.0}
            },
            "what_if": {
                "flag": whatif_lab,
                "is_counterfactual": is_whatif,
                "confidence": round(whatif_conf, 4),
            },
            "route_hints": route,
        }


if __name__ == "__main__":
    # sanity check kecil
    runner = ForecastMultiRunner(root_dir="./dummy")

    examples = [
        "nilai mesin XP888A 2 jam lagi berapa?",
        "coba prediksi semua mesin apakah akan ada anomali 2 jam kedepan?",
        "kapan XP888A bakal turun di bawah 70 derajat? kalau iya jam berapa pertama kalinya?",
        "andaikan data XP888A kemarin jam 10:00 gue ubah jadi 95, efeknya apa 2 jam lagi?",
        "ramalan performa semua mesin untuk 30 sampai 120 menit ke depan, interval 3 menit",
    ]
    for q in examples:
        print("\nQ:", q)
        print(runner.predict(q))
