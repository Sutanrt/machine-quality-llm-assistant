# ============================================================
# query_interpret.py
#
# Fungsi-fungsi untuk:
# - analisa teks natural language user (Bahasa Indonesia)
# - override hasil intent model (AnomalyQuadRunner.predict)
#   supaya lebih konsisten dengan aturan waktu / shift / dll
# - hasil akhirnya dipakai downstream (time parser, routing)
# ============================================================

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re


# -------------------------------------------------
# 1. Regex & heuristik dasar
# -------------------------------------------------

# Nama bulan Indonesia
MONTH_WORDS = (
    r"januari|februari|maret|april|mei|juni|"
    r"juli|agustus|september|oktober|november|desember"
)

# Rentang tanggal eksplisit (contoh: "10-20 Januari 2025")
DATE_RANGE_DAY = re.compile(
    rf"\b(\d{{1,2}})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{{1,2}})\s+({MONTH_WORDS})\s+(20\d{{2}})\b",
    re.I,
)

# Frasa relatif (X menit/jam/hari/minggu/bulan terakhir, minggu lalu, MTD/YTD/QTD)
REL_PATTERNS = [
    re.compile(r"\b(\d+)\s*(menit|jam|hari|minggu|bulan)\s+terakhir\b", re.I),
    re.compile(r"\b(minggu|bulan|tahun)\s+lalu\b", re.I),
    re.compile(r"\b(MTD|YTD|QTD)\b", re.I),
]

# Dayparts (pagi/siang/sore/malam/dini hari) → dianggap sebagai SHIFT
DAYPARTS = {
    "malam": re.compile(r"\bmalam\b", re.I),
    "pagi": re.compile(r"\bpagi\b", re.I),
    "siang": re.compile(r"\bsiang\b", re.I),
    "sore": re.compile(r"\bsore\b", re.I),
    "dini_hari": re.compile(r"\bdini\s*hari\b", re.I),
}

# Rentang jam/menit eksplisit: "jam 08:00 - 10:30", "8.00 s/d 10.00"
HOUR_MIN_RANGE = [
    re.compile(
        r"\b(jam|pukul)\s*(\d{1,2})[:\.]?(\d{2})?\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
        r"(\d{1,2})[:\.]?(\d{2})?\b",
        re.I,
    ),
    re.compile(
        r"\b(\d{1,2})[:\.](\d{2})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
        r"(\d{1,2})[:\.](\d{2})\b",
        re.I,
    ),
]

# Titik jam/menit tunggal: "jam 08:00", "pukul 9", "09.30"
HOUR_MIN_POINT = [
    re.compile(r"\b(jam|pukul)\s*(\d{1,2})[:\.]?(\d{2})?\b", re.I),
    re.compile(r"\b(\d{1,2})[:\.](\d{2})\b", re.I),
]

# Hint unit waktu:
DAY_HINT = re.compile(
    r"\bhari( ini|an)?\b|\bkemarin\b|\bsenin|selasa|rabu|kamis|jumat|jum’at|sabtu|minggu\b",
    re.I,
)
WEEK_HINT = re.compile(
    r"\bminggu(an)?\b|\bpekan\b|\bminggu\s+ke-\d+|\bminggu\s+pertama|\bminggu\s+kedua|\bminggu\s+ketiga|\bminggu\s+keempat",
    re.I,
)
MONTH_HINT = re.compile(
    rf"\b(bulan( ini|an| lalu)?|{MONTH_WORDS})\b",
    re.I,
)
YEAR_HINT = re.compile(
    r"\btahun( ini|an| lalu)?\b|\b20\d{2}\b",
    re.I,
)
SHIFT_WORD = re.compile(r"\bshift( pagi| siang| malam)?\b", re.I)


def any_machine_in_text(text: str) -> bool:
    """
    Heuristik deteksi ID mesin.
    Pola: huruf kapital + angka, misal 'XP888A', 'PQ667', dll.
    """
    return bool(re.search(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b", text))


# -------------------------------------------------
# 2. Routing hints cleanup
# -------------------------------------------------

def normalize_route(
    route_hints: List[str],
    gran_label: str,
    extras: Dict[str, Any],
) -> List[str]:
    """
    Membersihkan & menambah hint routing downstream.

    - Memastikan hanya slot_* extractor yang cocok dengan granularity akhir.
    - Menambah tag konteks "minute_range=present", "timeofday=malam", dll.
    - Preserve urutan kemunculan saat unique.
    """
    slots = {
        "slot_hour_extractor",
        "slot_minute_extractor",
        "slot_day_extractor",
        "slot_week_extractor",
        "slot_month_extractor",
        "slot_shift_extractor",
    }

    keep: List[str] = []

    # Pilih slot extractor utama berdasarkan granularity
    want = None
    if gran_label == "HOUR":
        want = (
            "slot_minute_extractor"
            if extras.get("minute_precision")
            else "slot_hour_extractor"
        )
    elif gran_label == "DAY":
        want = "slot_day_extractor"
    elif gran_label == "WEEK":
        want = "slot_week_extractor"
    elif gran_label == "MONTH":
        want = "slot_month_extractor"
    elif gran_label == "SHIFT":
        want = "slot_shift_extractor"

    if want:
        keep.append(want)

    # Pertahankan route_hints lain yang bukan slot_*
    for h in route_hints:
        if h not in slots:
            keep.append(h)

    # Tambahkan tag tambahan
    if gran_label == "HOUR":
        if extras.get("hour_range"):
            keep.append("minute_range=present")
        if extras.get("hour_point"):
            keep.append("minute_point=present")

    if gran_label == "SHIFT" and extras.get("timeofday"):
        keep.append(f"timeofday={extras['timeofday']}")

    # unique + preserve order
    return list(dict.fromkeys(keep))


# -------------------------------------------------
# 3. Aturan heuristik waktu berbasis teks mentah
# -------------------------------------------------

def infer_from_text(text: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Coba baca teks user langsung (tanpa model) untuk:
      - granularity target (HOUR/DAY/WEEK/MONTH/YEAR/SHIFT)
      - time_complexity target (RANGE_CLEAR / RANGE_REL / POINT / PERIOD / SHIFT)
      - extras (flag pendukung: hour_range, minute_precision, timeofday=malam, dst)
    """
    t = text.lower()
    extras: Dict[str, Any] = {}

    # 1) Rentang tanggal eksplisit: "10–20 Januari 2025"
    if DATE_RANGE_DAY.search(t):
        extras["date_range_day"] = True
        return "DAY", "RANGE_CLEAR", extras

    # 2) Range jam/menit eksplisit: "jam 08:00 - 10:30"
    for pat in HOUR_MIN_RANGE:
        if pat.search(t):
            extras["hour_range"] = True
            extras["minute_precision"] = True
            return "HOUR", "RANGE_CLEAR", extras

    # 3) Frasa relatif dalam menit/jam terakhir
    if re.search(r"\b(\d+)\s*menit\s+terakhir\b", t):
        extras["relative_minutes"] = True
        extras["minute_precision"] = True
        return "HOUR", "RANGE_REL", extras

    if re.search(r"\b(\d+)\s*jam\s+terakhir\b", t):
        extras["relative_hours"] = True
        return "HOUR", "RANGE_REL", extras

    # 4) Titik jam spesifik: "jam 08:00", "pukul 9"
    for pat in HOUR_MIN_POINT:
        if pat.search(t):
            extras["hour_point"] = True
            if re.search(r"[:\.]\d{2}", t):
                extras["minute_precision"] = True
            return "HOUR", "POINT", extras

    # 5) Daypart → dianggap SHIFT (contoh: "kemarin malam", "shift malam")
    for name, pat in DAYPARTS.items():
        if pat.search(t) or (name == "malam" and "kemarin malam" in t):
            extras["timeofday"] = name
            return "SHIFT", "SHIFT", extras

    # 6) Kata "shift" eksplisit
    if SHIFT_WORD.search(t):
        for name, pat in DAYPARTS.items():
            if pat.search(t):
                extras["timeofday"] = name
                break
        return "SHIFT", "SHIFT", extras

    # 7) Hint waktu umum
    if WEEK_HINT.search(t):
        return "WEEK", "PERIOD", extras
    if MONTH_HINT.search(t):
        return "MONTH", "PERIOD", extras
    if DAY_HINT.search(t):
        return "DAY", "PERIOD", extras
    if YEAR_HINT.search(t):
        return "YEAR", "PERIOD", extras

    return None, None, extras


def is_relative_range(text: str) -> bool:
    """
    True kalau ada frasa seperti:
    - "30 menit terakhir"
    - "7 hari terakhir"
    - "minggu lalu"
    - "MTD", "YTD"
    """
    return any(p.search(text) for p in REL_PATTERNS)


# -------------------------------------------------
# 4. Mapping "case" (A1.., R1..) sesuai scope/granularity
# -------------------------------------------------

def map_range_rel_case(scope_label: str, gran_label: str) -> str | None:
    """
    Range relatif ("X hari terakhir", "minggu lalu") → pilih R1..R6.
    Scope ALL pakai R4/R5/R6; SUBSET pakai R1/R2/R3.
    """
    g = (gran_label or "").upper()
    s = (scope_label or "").upper()
    if s == "ALL":
        return {"DAY": "R4", "WEEK": "R5", "MONTH": "R6"}.get(g)
    else:
        return {"DAY": "R1", "WEEK": "R2", "MONTH": "R3"}.get(g)


def map_period_case(scope_label: str, gran_label: str, model_case_label: str) -> str:
    """
    Periode agregat ("hari ini", "bulan ini", "minggu ini").
    Kalau ALL → A11/A12/A13.
    Kalau SUBSET → A2/A3/A4.
    Fallback ke label dari model kalau kombinasi tidak dikenali.
    """
    s = (scope_label or "").upper()
    g = (gran_label or "").upper()
    if s == "ALL":
        return {"DAY": "A11", "WEEK": "A12", "MONTH": "A13"}.get(g, model_case_label)
    else:
        return {"DAY": "A2", "WEEK": "A3", "MONTH": "A4"}.get(g, model_case_label)


def map_shift_case(scope_label: str) -> str:
    """SHIFT → A6 (subset) atau A15 (all)."""
    return "A15" if (scope_label or "").upper() == "ALL" else "A6"


def map_hour_case(scope_label: str, time_cplx: str, model_case_label: str) -> str:
    """
    Jam/menit granular.
    - RANGE_CLEAR  (rentang jam eksplisit)  → A14 / A5
    - RANGE_REL    (jam terakhir X jam)    → R4 / R1
    - POINT        (jam spesifik)          → fallback ke model_case_label
    """
    s = (scope_label or "").upper()
    if time_cplx == "RANGE_CLEAR":
        return "A14" if s == "ALL" else "A5"
    if time_cplx == "RANGE_REL":
        return "R4" if s == "ALL" else "R1"
    return model_case_label


# -------------------------------------------------
# 5. Fungsi utama: smart_predict
# -------------------------------------------------

def smart_predict(
    text: str,
    runner,
    conf_floor: float = 0.50,
) -> Dict[str, Any]:
    """
    Post-process hasil intent model (runner.predict)
    supaya:
    - scope 'ALL' kalau user gak sebut mesin sama sekali
    - granularity/time_complexity disesuaikan rules linguistik
    - case (Axx / Rxx) disesuaikan granularity & scope
    - route_hints dirapikan

    Input:
        text   : query user (Bahasa Indonesia)
        runner : instance AnomalyQuadRunner (punya .predict(str) -> dict)
        conf_floor : threshold confidence granularity, di bawah ini kita boleh override pakai regex hint

    Output dict:
    {
        "scope": {"label": ..., "confidence": ...},
        "time_granularity": {"label": ..., "confidence": ...},
        "time_complexity": {"label": ..., "confidence": ...},
        "case": {"label": ..., "confidence": ...},
        "route_hints": [...],
        "postprocess_reason": [...],
    }
    """

    raw = runner.predict(text)

    scope = raw["scope"]["label"]
    gran = raw["time_granularity"]["label"]
    gran_conf = raw["time_granularity"]["confidence"]
    cplx = raw["time_complexity"]["label"]
    case = raw["case"]["label"]
    route = list(raw.get("route_hints", []))
    reasons: List[str] = []

    # --- (1) Scope fix:
    # Kalau user gak nyebut mesin apa pun, tapi model bilang SUBSET/NOT_ALL,
    # kita paksa jadi ALL.
    if not any_machine_in_text(text) and scope == "NOT_ALL":
        scope = "ALL"
        reasons.append("scope fix: no machine entity -> ALL")

    # --- (2) Heuristik bahasa alami: prioritaskan ekspresi eksplisit
    g2, x2, extras = infer_from_text(text)

    # granularity override?
    if g2 and g2 != gran:
        reasons.append(f"gran {gran}->{g2} (rules)")
        gran = g2

    # time_complexity override?
    if x2 and x2 != cplx:
        reasons.append(f"cplx {cplx}->{x2} (rules)")
        cplx = x2

    # Jika ada rentang jam/tanggal eksplisit → lock ke RANGE_CLEAR
    if extras.get("hour_range") or extras.get("date_range_day"):
        cplx = "RANGE_CLEAR"
        reasons.append("cplx lock: explicit time range -> RANGE_CLEAR")

    # Frasa relatif (misal "terakhir", "minggu lalu") → RANGE_REL
    if (
        is_relative_range(text)
        and not (extras.get("hour_range") or extras.get("date_range_day"))
        and cplx != "RANGE_REL"
    ):
        cplx = "RANGE_REL"
        reasons.append("cplx -> RANGE_REL (relative phrase)")

    # --- (3) Low-confidence fallback untuk granularity
    if gran_conf < conf_floor and not g2:
        t = text.lower()
        if WEEK_HINT.search(t) and gran != "WEEK":
            reasons.append(f"gran lowconf {gran_conf}->WEEK")
            gran = "WEEK"
        elif MONTH_HINT.search(t) and gran != "MONTH":
            reasons.append(f"gran lowconf {gran_conf}->MONTH")
            gran = "MONTH"
        elif DAY_HINT.search(t) and gran != "DAY":
            reasons.append(f"gran lowconf {gran_conf}->DAY")
            gran = "DAY"

    # --- (4) map 'case' ke kode final konsisten
    if gran == "SHIFT":
        case2 = map_shift_case(scope)
    elif gran == "HOUR":
        case2 = map_hour_case(scope, cplx, case)
    elif cplx == "RANGE_REL" and gran in {"DAY", "WEEK", "MONTH"}:
        case2 = map_range_rel_case(scope, gran)
    elif cplx == "PERIOD" and gran in {"DAY", "WEEK", "MONTH"}:
        case2 = map_period_case(scope, gran, case)
    else:
        case2 = case

    if case2 and case2 != case:
        reasons.append(f"case {case}->{case2}")
        case = case2

    # --- (5) route_hints disinkronkan dengan granularity final
    route = normalize_route(route, gran, extras)

    # --- (6) final output
    return {
        "scope": {
            "label": scope,
            "confidence": raw["scope"]["confidence"],
        },
        "time_granularity": {
            "label": gran,
            "confidence": raw["time_granularity"]["confidence"],
        },
        "time_complexity": {
            "label": cplx,
            "confidence": raw["time_complexity"]["confidence"],
        },
        "case": {
            "label": case,
            "confidence": raw["case"]["confidence"],
        },
        "route_hints": route,
        "postprocess_reason": reasons,
    }

# =======================================================================================
# Cara Pakai
# from intent_runtime import AnomalyQuadRunner
# from query_interpret import smart_predict
# from time_parse import resolve_time_window
# from anomaly_query_engine import run_anomaly_query
# from other_ts_loader import load_timeseries_from_df_raw2

# runner = AnomalyQuadRunner(
#     scope_dir="runs_anomaly_quad_relclass/scope/final_model",
#     gran_dir="runs_anomaly_quad_relclass/time_granularity/final_model",
#     cplx_dir="runs_anomaly_quad_relclass/time_complexity/final_model",
#     case_dir="runs_anomaly_quad_relclass/case/final_model",
# )

# query = "apakah ada anomali kemarin malam?"
# pred  = smart_predict(query, runner)
# tw    = resolve_time_window(query, pred)

# df_ts = load_timeseries_from_df_raw2("/workspace/df_raw2.csv", assume_us_datetime=True)
# df_thr = ... # load threshold min/max kamu

# res = run_anomaly_query(
#     text=query,
#     runner=runner,                     # masih dipakai di run_anomaly_query untuk pred full
#     resolve_time_window_fn=resolve_time_window,
#     df_ts=df_ts,
#     df_thr=df_thr,
# )
# =======================================================================================

# import re

# # --- regex bulan Indonesia & tanggal rentang ---
# MONTH_WORDS = r"januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember"
# DATE_RANGE_DAY = re.compile(
#     rf"\b(\d{{1,2}})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{{1,2}})\s+({MONTH_WORDS})\s+(20\d{{2}})\b",
#     re.I
# )

# # --- relative phrases (jam/menit/hari/minggu/bulan) ---
# REL_PATTERNS = [
#     re.compile(r"\b(\d+)\s*(menit|jam|hari|minggu|bulan)\s+terakhir\b", re.I),
#     re.compile(r"\b(minggu|bulan|tahun)\s+lalu\b", re.I),
#     re.compile(r"\b(MTD|YTD|QTD)\b", re.I),
# ]

# DAYPARTS = {
#     "malam": re.compile(r"\bmalam\b", re.I),
#     "pagi": re.compile(r"\bpagi\b", re.I),
#     "siang": re.compile(r"\bsiang\b", re.I),
#     "sore": re.compile(r"\bsore\b", re.I),
#     "dini_hari": re.compile(r"\bdini\s*hari\b", re.I),
# }

# # jam/menit
# HOUR_MIN_RANGE = [
#     re.compile(r"\b(jam|pukul)\s*(\d{1,2})[:\.]?(\d{2})?\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{1,2})[:\.]?(\d{2})?\b", re.I),
#     re.compile(r"\b(\d{1,2})[:\.](\d{2})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{1,2})[:\.](\d{2})\b", re.I),
# ]
# HOUR_MIN_POINT = [
#     re.compile(r"\b(jam|pukul)\s*(\d{1,2})[:\.]?(\d{2})?\b", re.I),
#     re.compile(r"\b(\d{1,2})[:\.](\d{2})\b", re.I),
# ]

# # unit hints
# DAY_HINT   = re.compile(r"\bhari( ini|an)?\b|\bkemarin\b|\bsenin|selasa|rabu|kamis|jumat|jum’at|sabtu|minggu\b", re.I)
# WEEK_HINT  = re.compile(r"\bminggu(an)?\b|\bpekan\b|\bminggu\s+ke-\d+|\bminggu\s+pertama|\bminggu\s+kedua|\bminggu\s+ketiga|\bminggu\s+keempat", re.I)
# MONTH_HINT = re.compile(rf"\b(bulan( ini|an| lalu)?|{MONTH_WORDS})\b", re.I)
# YEAR_HINT  = re.compile(r"\btahun( ini|an| lalu)?\b|\b20\d{2}\b", re.I)
# SHIFT_WORD = re.compile(r"\bshift( pagi| siang| malam)?\b", re.I)

# def any_machine_in_text(text):
#     # heuristik: ada KAPITAL+digit (XP888A, PQ667, dsb.)
#     return bool(re.search(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b", text))

# def normalize_route(route_hints, gran_label, extras):
#     """Pastikan hanya slot_* yang sesuai granularity, tambah tag menit/daypart dengan benar."""
#     slots = {"slot_hour_extractor","slot_minute_extractor","slot_day_extractor","slot_week_extractor","slot_month_extractor","slot_shift_extractor"}
#     keep = []
#     # pilih slot sesuai gran
#     want = None
#     if gran_label=="HOUR":
#         want = "slot_minute_extractor" if extras.get("minute_precision") else "slot_hour_extractor"
#     elif gran_label=="DAY":   want = "slot_day_extractor"
#     elif gran_label=="WEEK":  want = "slot_week_extractor"
#     elif gran_label=="MONTH": want = "slot_month_extractor"
#     elif gran_label=="SHIFT": want = "slot_shift_extractor"

#     if want:
#         keep.append(want)

#     # non-slot hints pertahankan
#     for h in route_hints:
#         if h not in slots:
#             keep.append(h)

#     # minute/daypart tags
#     if gran_label=="HOUR":
#         if extras.get("hour_range"): keep.append("minute_range=present")
#         if extras.get("hour_point"): keep.append("minute_point=present")
#     if gran_label=="SHIFT" and extras.get("timeofday"):
#         keep.append(f"timeofday={extras['timeofday']}")

#     return list(dict.fromkeys(keep))  # unique, keep order

# def infer_from_text(text: str):
#     t = text.lower()
#     extras = {}

#     # 1) Rentang tanggal eksplisit: 10–20 Januari 2025
#     if DATE_RANGE_DAY.search(t):
#         extras["date_range_day"] = True
#         return "DAY", "RANGE_CLEAR", extras

#     # 2) Range jam/menit
#     for pat in HOUR_MIN_RANGE:
#         if pat.search(t):
#             extras["hour_range"] = True
#             extras["minute_precision"] = True
#             return "HOUR", "RANGE_CLEAR", extras

#     # 3) Relative minutes/hours
#     if re.search(r"\b(\d+)\s*menit\s+terakhir\b", t):
#         extras["relative_minutes"] = True; extras["minute_precision"] = True
#         return "HOUR", "RANGE_REL", extras
#     if re.search(r"\b(\d+)\s*jam\s+terakhir\b", t):
#         extras["relative_hours"] = True
#         return "HOUR", "RANGE_REL", extras

#     # 4) Titik jam/menit
#     for pat in HOUR_MIN_POINT:
#         if pat.search(t):
#             extras["hour_point"] = True
#             if re.search(r"[:\.]\d{2}", t): extras["minute_precision"] = True
#             return "HOUR", "POINT", extras

#     # 5) Daypart → SHIFT (termasuk “kemarin malam”)
#     for name, pat in DAYPARTS.items():
#         if pat.search(t) or (name=="malam" and "kemarin malam" in t):
#             extras["timeofday"] = name
#             return "SHIFT", "SHIFT", extras

#     # 6) Kata "shift"
#     if SHIFT_WORD.search(t):
#         for name, pat in DAYPARTS.items():
#             if pat.search(t): extras["timeofday"] = name; break
#         return "SHIFT", "SHIFT", extras

#     # 7) Umum DAY/WEEK/MONTH/YEAR
#     if WEEK_HINT.search(t):  return "WEEK",  "PERIOD", extras
#     if MONTH_HINT.search(t): return "MONTH", "PERIOD", extras
#     if DAY_HINT.search(t):   return "DAY",   "PERIOD", extras
#     if YEAR_HINT.search(t):  return "YEAR",  "PERIOD", extras
#     return None, None, extras

# def is_relative_range(text: str):
#     return any(p.search(text) for p in REL_PATTERNS)

# def map_range_rel_case(scope_label: str, gran_label: str):
#     g = (gran_label or "").upper(); s = (scope_label or "").upper()
#     if s=="ALL":  return {"DAY":"R4","WEEK":"R5","MONTH":"R6"}.get(g)
#     else:         return {"DAY":"R1","WEEK":"R2","MONTH":"R3"}.get(g)

# def map_period_case(scope_label: str, gran_label: str, model_case_label: str):
#     s = (scope_label or "").upper(); g = (gran_label or "").upper()
#     if s=="ALL":  return {"DAY":"A11","WEEK":"A12","MONTH":"A13"}.get(g, model_case_label)
#     else:         return {"DAY":"A2","WEEK":"A3","MONTH":"A4"}.get(g, model_case_label)

# def map_shift_case(scope_label: str):   return "A15" if (scope_label or "").upper()=="ALL" else "A6"
# def map_hour_case(scope_label: str, time_cplx: str, model_case_label: str):
#     s = (scope_label or "").upper()
#     if time_cplx=="RANGE_CLEAR": return "A14" if s=="ALL" else "A5"
#     if time_cplx=="RANGE_REL":   return "R4" if s=="ALL" else "R1"
#     return model_case_label

# def smart_predict(text: str, runner, conf_floor=0.50):
#     raw = runner.predict(text)
#     scope = raw["scope"]["label"]
#     gran  = raw["time_granularity"]["label"]; gran_conf = raw["time_granularity"]["confidence"]
#     cplx  = raw["time_complexity"]["label"]
#     case  = raw["case"]["label"]
#     route = list(raw.get("route_hints", []))
#     reasons = []

#     # Scope fix: jika tidak ada entity mesin sama sekali → anggap ALL
#     if not any_machine_in_text(text) and scope=="NOT_ALL":
#         scope = "ALL"; reasons.append("scope fix: no machine entity -> ALL")

#     # Infer dari teks (prioritas: date range/daypart/hour/minute > relative > period)
#     g2, x2, extras = infer_from_text(text)
#     if g2 and g2 != gran: reasons.append(f"gran {gran}->{g2} (rules)"); gran = g2
#     if x2 and x2 != cplx: reasons.append(f"cplx {cplx}->{x2} (rules)"); cplx = x2

#     # Jika ada hour/minute/date range eksplisit → **kunci** ke RANGE_CLEAR (jangan diubah oleh frasa relatif)
#     if extras.get("hour_range") or extras.get("date_range_day"):
#         cplx = "RANGE_CLEAR"; reasons.append("cplx lock: explicit time range -> RANGE_CLEAR")

#     # Frasa relatif aktif jika tidak ada lock explicit range
#     if is_relative_range(text) and not (extras.get("hour_range") or extras.get("date_range_day")) and cplx!="RANGE_REL":
#         cplx = "RANGE_REL"; reasons.append("cplx -> RANGE_REL (relative phrase)")

#     # Low-conf gran fallback
#     if gran_conf < conf_floor and not g2:
#         t = text.lower()
#         if WEEK_HINT.search(t)  and gran!="WEEK":  reasons.append(f"gran lowconf {gran_conf}->WEEK");  gran="WEEK"
#         elif MONTH_HINT.search(t) and gran!="MONTH": reasons.append(f"gran lowconf {gran_conf}->MONTH"); gran="MONTH"
#         elif DAY_HINT.search(t)   and gran!="DAY":   reasons.append(f"gran lowconf {gran_conf}->DAY");   gran="DAY"

#     # Case mapping konsisten
#     if gran=="SHIFT": case2 = map_shift_case(scope); 
#     elif gran=="HOUR": case2 = map_hour_case(scope, cplx, case)
#     elif cplx=="RANGE_REL" and gran in {"DAY","WEEK","MONTH"}: case2 = map_range_rel_case(scope, gran)
#     elif cplx=="PERIOD"    and gran in {"DAY","WEEK","MONTH"}: case2 = map_period_case(scope, gran, case)
#     else: case2 = case
#     if case2 and case2 != case: reasons.append(f"case {case}->{case2}"); case = case2

#     # Sinkronisasi route_hints
#     route = normalize_route(route, gran, extras)

#     return {
#         "scope": {"label": scope, "confidence": raw["scope"]["confidence"]},
#         "time_granularity": {"label": gran, "confidence": raw["time_granularity"]["confidence"]},
#         "time_complexity": {"label": cplx, "confidence": raw["time_complexity"]["confidence"]},
#         "case": {"label": case, "confidence": raw["case"]["confidence"]},
#         "route_hints": route,
#         "postprocess_reason": reasons
#     }
