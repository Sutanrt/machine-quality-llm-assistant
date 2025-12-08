from __future__ import annotations

import re
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any


# =========================
# Config
# =========================

# Anggap "hari ini" untuk parser waktu.
# Di production kamu bisa ganti ini ke datetime.now(tz=Asia/Jakarta)
NOW = datetime(2025, 10, 23)  # Asia/Jakarta "today"


# =========================
# Static dictionaries
# =========================

ID_MONTHS = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}

ID_DAYS = {
    "senin": 0,
    "selasa": 1,
    "rabu": 2,
    "kamis": 3,
    "jumat": 4,
    "jum’at": 4,
    "sabtu": 5,
    "minggu": 6,
    "ahad": 6,
}


# =========================
# Low-level time helpers
# =========================

def month_range(year: int, month: int):
    """Return (start_of_month, end_of_month_23_59_59)."""
    last_day = calendar.monthrange(year, month)[1]
    return (
        datetime(year, month, 1, 0, 0, 0),
        datetime(year, month, last_day, 23, 59, 59),
    )


def year_range(year: int):
    """Return (start_of_year, end_of_year_23_59_59)."""
    return (
        datetime(year, 1, 1, 0, 0, 0),
        datetime(year, 12, 31, 23, 59, 59),
    )


def prev_weekday(target_weekday: int, ref: datetime = NOW):
    """
    Ambil hari terakhir (<= ref) yang weekday-nya = target_weekday.
    Senin=0 ... Minggu=6.
    """
    delta = (ref.weekday() - target_weekday) % 7
    day_start = (ref - timedelta(days=delta)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end = (ref - timedelta(days=delta)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    return day_start, day_end


def end_of_day(dt: datetime) -> datetime:
    """Return dt's same calendar day but at 23:59:59."""
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


# =========================
# Machine extractor
# =========================

MACHINE_PAT = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b")

def extract_machines(text: str) -> List[str]:
    """
    Ekstrak kode mesin seperti XP888A, PQ667.
    Dedup tapi tetap jaga urutan kemunculan.
    """
    seen, out = set(), []
    for m in MACHINE_PAT.findall(text.upper()):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# =========================
# TimeExtraction dataclass
# =========================

@dataclass
class TimeExtraction:
    """
    kind:
      - "RANGE"  : rentang multi-bulan/tahun yang eksplisit ("november sampai desember tahun lalu")
      - "PERIOD" : 1 periode kalender ("bulan ini", "bulan lalu", "2024")
      - "POINT"  : 1 hari spesifik ("senin", "kemarin", "hari ini")
    granularity:
      - "HOUR", "DAY", "WEEK", "MONTH", "YEAR", "ALL"
    """
    kind: str
    start: Optional[datetime]
    end: Optional[datetime]
    granularity: Optional[str]
    note: Optional[str] = None


# =========================
# Natural-language time parser
# =========================

def parse_time_id(text: str) -> Optional[TimeExtraction]:
    """
    Parse frasa waktu Bahasa Indonesia menjadi TimeExtraction.
    Contoh:
    - "november sampai desember tahun lalu"
    - "bulan ini"
    - "kemarin"
    - "tahun 2024"
    """
    t = text.lower().strip()

    # 1) Rentang bulan: "november sampai desember tahun lalu"
    m_range = re.search(
        r"\b(" + "|".join(ID_MONTHS.keys()) + r")\b.*?\b(s/d|sd|sampai|hingga|-)\b.*?\b("
        + "|".join(ID_MONTHS.keys()) + r")\b(?:\s+(tahun\s+lalu|tahun\s+ini|\d{4}))?",
        t,
    )
    if m_range:
        m1 = ID_MONTHS[m_range.group(1)]
        m2 = ID_MONTHS[m_range.group(3)]
        tail = (m_range.group(4) or "").strip()

        if tail == "tahun lalu":
            y = NOW.year - 1
        elif tail == "tahun ini":
            y = NOW.year
        elif tail.isdigit() and len(tail) == 4:
            y = int(tail)
        else:
            # default kalau gak jelas: cek "tahun lalu" di kalimat atau pakai tahun ini
            y = NOW.year - 1 if "tahun lalu" in t else NOW.year

        s1, e1 = month_range(y, m1)
        s2, e2 = month_range(y, m2)

        # rentang dari bulan lebih kecil -> bulan lebih besar (robust)
        s = s1 if s1 < s2 else s2
        e = e1 if e1 > e2 else e2

        # kalau rentang tahun ini, batasi end sampai EOD hari ini
        if y == NOW.year:
            e = min(e, end_of_day(NOW))

        return TimeExtraction(
            kind="RANGE",
            start=s,
            end=e,
            granularity="MONTH",
            note="range bulan",
        )

    # 2) Satu bulan spesifik: "april tahun lalu"
    m_single = re.search(
        r"\b(" + "|".join(ID_MONTHS.keys()) + r")\b(?:\s+(tahun\s+lalu|tahun\s+ini|\d{4}))?",
        t,
    )
    if m_single:
        m = ID_MONTHS[m_single.group(1)]
        tail = (m_single.group(2) or "").strip()

        if tail == "tahun lalu":
            y = NOW.year - 1
        elif tail == "tahun ini":
            y = NOW.year
        elif tail.isdigit() and len(tail) == 4:
            y = int(tail)
        else:
            y = NOW.year - 1 if "tahun lalu" in t else NOW.year

        s, e = month_range(y, m)
        if y == NOW.year:
            # month-to-date
            e = min(e, end_of_day(NOW))

        return TimeExtraction(
            kind="PERIOD",
            start=s,
            end=e,
            granularity="MONTH",
            note="bulan tunggal",
        )

    # 3) "bulan ini" / "bulan lalu"
    if "bulan ini" in t:
        s, e = month_range(NOW.year, NOW.month)
        e = min(e, end_of_day(NOW))  # month-to-date
        return TimeExtraction(
            kind="PERIOD",
            start=s,
            end=e,
            granularity="MONTH",
            note="bulan ini (MTD)",
        )

    if "bulan lalu" in t:
        y = NOW.year
        m = NOW.month - 1
        if m == 0:
            y -= 1
            m = 12
        s, e = month_range(y, m)
        return TimeExtraction(
            kind="PERIOD",
            start=s,
            end=e,
            granularity="MONTH",
            note="bulan lalu",
        )

    # 4) Nama hari: "senin", "rabu"
    for name, wd in ID_DAYS.items():
        if re.search(rf"\b{name}\b", t):
            s, e = prev_weekday(wd, NOW)
            return TimeExtraction(
                kind="POINT",
                start=s,
                end=e,
                granularity="DAY",
                note=f"hari {name}",
            )

    # 5) "hari ini" / "kemarin"
    if "hari ini" in t:
        day_start = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        return TimeExtraction(
            kind="POINT",
            start=day_start,
            end=NOW,  # sampai waktu sekarang
            granularity="DAY",
            note="hari ini",
        )

    if "kemarin" in t:
        d = NOW - timedelta(days=1)
        s = d.replace(hour=0, minute=0, second=0, microsecond=0)
        e = d.replace(hour=23, minute=59, second=59, microsecond=0)
        return TimeExtraction(
            kind="POINT",
            start=s,
            end=e,
            granularity="DAY",
            note="kemarin",
        )

    # 6) Tahun: "tahun lalu", "tahun ini", "tahun 2024"
    if "tahun lalu" in t:
        y = NOW.year - 1
        s, e = year_range(y)
        return TimeExtraction(
            kind="RANGE",
            start=s,
            end=e,
            granularity="YEAR",
            note="tahun lalu",
        )

    if "tahun ini" in t:
        y = NOW.year
        s, e = year_range(y)
        e = min(e, end_of_day(NOW))  # year-to-date
        return TimeExtraction(
            kind="PERIOD",
            start=s,
            end=e,
            granularity="YEAR",
            note="tahun ini (YTD)",
        )

    y_exp = re.search(r"tahun\s+(\d{4})", t)
    if y_exp:
        y = int(y_exp.group(1))
        s, e = year_range(y)
        if y == NOW.year:
            e = min(e, end_of_day(NOW))
        return TimeExtraction(
            kind="PERIOD",
            start=s,
            end=e,
            granularity="YEAR",
            note="tahun eksplisit",
        )

    # kalau gak bisa parse apa pun
    return None


# =========================
# High-level intent merger
# =========================

def interpret(query_text: str, model_pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gabungkan:
    - hasil classifier model_pred (scope/time/case)
    - hasil rule-based parser (waktu bahasa manusia, machine_id)
    -> intent final yang dipakai downstream.
    """

    time_info = parse_time_id(query_text)  # TimeExtraction | None
    machines = extract_machines(query_text)

    # granularity gabungan: parser menang kalau punya jawaban
    if time_info:
        final_gran = time_info.granularity or model_pred["time"]["label"]
        start = time_info.start.isoformat() if time_info.start else None
        end = time_info.end.isoformat() if time_info.end else None
        note = time_info.note
    else:
        final_gran = model_pred["time"]["label"]
        start = end = note = None

    # scope: ALL vs SUBSET
    scope_label = model_pred["scope"]["label"]
    scope = "ALL" if scope_label == "ALL" else "SUBSET"

    # petunjuk routing (bisa dipakai engine di step berikutnya)
    route_hints = []
    if time_info and time_info.kind == "RANGE":
        route_hints.append("use_range_extractor")
    if final_gran in {"HOUR", "DAY", "WEEK", "MONTH", "YEAR"}:
        route_hints.append(f"slot_{final_gran.lower()}_extractor")
    route_hints.append(
        "aggregate_all_machines" if scope == "ALL" else "machine_entity_extractor"
    )

    # refine case (SINGLE_POINT vs SINGLE_PERIOD vs COMPARE, dll)
    qlow = query_text.lower()
    case_label = model_pred["case"]["label"]

    if time_info and time_info.kind == "POINT":
        case_label = "SINGLE_POINT"

    if time_info and time_info.kind == "PERIOD":
        # kecuali user jelas minta perbandingan
        if (
            "ringkasan" not in qlow
            and "ringkas" not in qlow
            and case_label != "COMPARE"
        ):
            case_label = "SINGLE_PERIOD"

    # Build final dict
    intent = {
        "query": query_text,
        "scope": scope,
        "machines": machines if scope == "SUBSET" else [],
        "granularity": final_gran,
        "time_range": {
            "start": start,
            "end": end,
            "note": note,
        },
        "case": case_label,
        "route_hints": route_hints,
        "meta": {
            "model_time_label": model_pred["time"]["label"],
            "model_time_conf": model_pred["time"]["confidence"],
            "model_case_conf": model_pred["case"]["confidence"],
            "model_scope_conf": model_pred["scope"]["confidence"],
            "extractor_kind": time_info.kind if time_info else None,
        },
    }

    return intent
# ======================================================================================
# CARA PAKAI
# from nlp_intent import interpret, extract_machines, parse_time_id
# from trend_analysis import analyze_trend, pretty_response, extract_subintent

# pred = runner.predict(q)
# intent = interpret(q, pred)
# sub   = extract_subintent(q)
# result = analyze_trend(df_raw, intent, subintent=sub)
# print(pretty_response(result, sub))
# ======================================================================================


# # =========================================================
# # Indonesian Time & Entity Extractor (for Triple-Head output)
# # - Converts phrases like "november sampai desember tahun lalu",
# #   "senin", "bulan ini", "tahun lalu" into absolute date ranges
# # - Extracts machine codes e.g. XP888A, PQ667
# # - Merges with model predictions (scope/time/case/route_hints)
# #   into a structured intent dict
# # Assumptions:
# #   - Timezone: Asia/Jakarta
# #   - "today" is 2025-10-23 (can be changed via NOW)
# # =========================================================
# from __future__ import annotations
# import re
# from dataclasses import dataclass, asdict
# from datetime import datetime, timedelta
# import calendar

# # ---- Config ----
# NOW = datetime(2025, 10, 23)  # Asia/Jakarta today; ganti jika diperlukan

# # ---- Helpers ----
# ID_MONTHS = {
#     "januari":1, "februari":2, "maret":3, "april":4, "mei":5, "juni":6,
#     "juli":7, "agustus":8, "september":9, "oktober":10, "november":11, "desember":12
# }
# ID_DAYS = {
#     "senin":0, "selasa":1, "rabu":2, "kamis":3, "jumat":4, "jum’at":4, "sabtu":5, "minggu":6, "ahad":6
# }
# def month_range(year:int, month:int):
#     last_day = calendar.monthrange(year, month)[1]
#     return datetime(year, month, 1), datetime(year, month, last_day, 23, 59, 59)

# def year_range(year:int):
#     return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)

# def clip_to_now(start:datetime, end:datetime, clip_to_today:bool=True):
#     if clip_to_today and end > NOW:
#         end = NOW
#     return start, end

# def prev_weekday(target_weekday:int, ref:datetime=NOW):
#     # Return most recent <= ref day that matches target_weekday (Mon=0..Sun=6)
#     delta = (ref.weekday() - target_weekday) % 7
#     return (ref - timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0), \
#            (ref - timedelta(days=delta)).replace(hour=23, minute=59, second=59, microsecond=0)

# # ---- Entity extractor (simple) ----
# MACHINE_PAT = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b")

# def extract_machines(text:str):
#     # dedup while preserving order
#     seen, out = set(), []
#     for m in MACHINE_PAT.findall(text.upper()):
#         if m not in seen:
#             seen.add(m); out.append(m)
#     return out

# # ---- Time extractor ----
# @dataclass
# class TimeExtraction:
#     kind: str              # "RANGE","PERIOD","POINT"
#     start: datetime | None
#     end: datetime | None
#     granularity: str | None     # "HOUR/DAY/WEEK/MONTH/YEAR/ALL"
#     note: str | None = None     # any heuristic note

# # --- helper baru: end-of-day untuk NOW ---
# def end_of_day(dt: datetime) -> datetime:
#     return dt.replace(hour=23, minute=59, second=59, microsecond=0)

# # --- perbaikan: urutan prioritas + EOD handling ---
# def parse_time_id(text:str) -> TimeExtraction | None:
#     t = text.lower().strip()

#     # 1) RANGE BULAN (prioritas tertinggi)  e.g. "november sampai desember (tahun lalu/ini/2024)"
#     m_range = re.search(
#         r"\b(" + "|".join(ID_MONTHS.keys()) + r")\b.*?\b(s/d|sd|sampai|hingga|-)\b.*?\b(" +
#         "|".join(ID_MONTHS.keys()) + r")\b(?:\s+(tahun\s+lalu|tahun\s+ini|\d{4}))?",
#         t
#     )
#     if m_range:
#         m1 = ID_MONTHS[m_range.group(1)]
#         m2 = ID_MONTHS[m_range.group(3)]
#         tail = (m_range.group(4) or "").strip()
#         if tail == "tahun lalu":
#             y = NOW.year - 1
#         elif tail == "tahun ini":
#             y = NOW.year
#         elif tail.isdigit() and len(tail)==4:
#             y = int(tail)
#         else:
#             y = NOW.year - 1 if "tahun lalu" in t else NOW.year
#         s1,e1 = month_range(y, m1)
#         s2,e2 = month_range(y, m2)
#         s, e = (s1 if s1 < s2 else s2), (e1 if e1 > e2 else e2)
#         if y == NOW.year:
#             # kalau tahun ini, batasi sampai EOD hari ini
#             e = min(e, end_of_day(NOW))
#         return TimeExtraction("RANGE", s, e, "MONTH", "range bulan")

#     # 2) BULAN TUNGGAL (april [tahun lalu/ini/2024])
#     m_single = re.search(
#         r"\b(" + "|".join(ID_MONTHS.keys()) + r")\b(?:\s+(tahun\s+lalu|tahun\s+ini|\d{4}))?", t
#     )
#     if m_single:
#         m = ID_MONTHS[m_single.group(1)]
#         tail = (m_single.group(2) or "").strip()
#         if tail == "tahun lalu":
#             y = NOW.year - 1
#         elif tail == "tahun ini":
#             y = NOW.year
#         elif tail.isdigit() and len(tail)==4:
#             y = int(tail)
#         else:
#             y = NOW.year - 1 if "tahun lalu" in t else NOW.year
#         s,e = month_range(y, m)
#         if y == NOW.year:
#             e = min(e, end_of_day(NOW))  # MTD
#         return TimeExtraction("PERIOD", s, e, "MONTH", "bulan tunggal")

#     # 3) BULAN INI / BULAN LALU (PERIOD)
#     if "bulan ini" in t:
#         s,e = month_range(NOW.year, NOW.month)
#         e = min(e, end_of_day(NOW))     # MTD sampai akhir hari ini
#         return TimeExtraction("PERIOD", s, e, "MONTH", "bulan ini (MTD)")

#     if "bulan lalu" in t:
#         y = NOW.year; m = NOW.month - 1
#         if m == 0: y -= 1; m = 12
#         s,e = month_range(y, m)
#         return TimeExtraction("PERIOD", s, e, "MONTH", "bulan lalu")

#     # 4) HARI DALAM MINGGU ("senin", "rabu", ...)
#     for name, wd in ID_DAYS.items():
#         if re.search(rf"\b{name}\b", t):
#             s,e = prev_weekday(wd, NOW)
#             return TimeExtraction("POINT", s, e, "DAY", f"hari {name}")

#     # 5) HARI INI / KEMARIN
#     if "hari ini" in t:
#         s = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
#         e = NOW  # waktu sekarang; bisa juga end_of_day(NOW) jika maunya EOD
#         return TimeExtraction("POINT", s, e, "DAY", "hari ini")

#     if "kemarin" in t:
#         d = NOW - timedelta(days=1)
#         s = d.replace(hour=0, minute=0, second=0, microsecond=0)
#         e = d.replace(hour=23, minute=59, second=59, microsecond=0)
#         return TimeExtraction("POINT", s, e, "DAY", "kemarin")

#     # 6) TAHUN LALU / TAHUN INI / TAHUN YYYY  (fallback tahunan)
#     if "tahun lalu" in t:
#         y = NOW.year - 1
#         s,e = year_range(y)
#         return TimeExtraction("RANGE", s, e, "YEAR", "tahun lalu")

#     if "tahun ini" in t:
#         y = NOW.year
#         s,e = year_range(y)
#         e = min(e, end_of_day(NOW))  # YTD
#         return TimeExtraction("PERIOD", s, e, "YEAR", "tahun ini (YTD)")

#     y_exp = re.search(r"tahun\s+(\d{4})", t)
#     if y_exp:
#         y = int(y_exp.group(1))
#         s,e = year_range(y)
#         if y == NOW.year:
#             e = min(e, end_of_day(NOW))
#         return TimeExtraction("PERIOD", s, e, "YEAR", "tahun eksplisit")

#     # Tidak ditemukan
#     return None

# # --- interpret(): hanya ubah bagian route range flag ---
# def interpret(query_text:str, model_pred:dict):
#     time_info = parse_time_id(query_text)  # may be None
#     machines = extract_machines(query_text)

#     # final granularity from extractor (if present)
#     if time_info:
#         final_gran = time_info.granularity
#         start = time_info.start.isoformat() if time_info.start else None
#         end   = time_info.end.isoformat()   if time_info.end   else None
#         note  = time_info.note
#     else:
#         final_gran = model_pred["time"]["label"]
#         start = end = note = None

#     scope_label = model_pred["scope"]["label"]
#     scope = "ALL" if scope_label == "ALL" else "SUBSET"

#     # route_hints (extractor-first)
#     route = []
#     if time_info and time_info.kind == "RANGE":
#         route.append("use_range_extractor")
#     if final_gran in {"HOUR","DAY","WEEK","MONTH","YEAR"}:
#         route.append(f"slot_{final_gran.lower()}_extractor")
#     route.append("aggregate_all_machines" if scope == "ALL" else "machine_entity_extractor")

#     # ----- CASE FIX -----
#     qlow = query_text.lower()
#     case_label = model_pred["case"]["label"]
#     # titik waktu tunggal -> SINGLE_POINT
#     if time_info and time_info.kind == "POINT":
#         case_label = "SINGLE_POINT"
#     # periode tunggal (bulan ini/lalu, bulan eksplisit, tahun ini) -> SINGLE_PERIOD
#     if time_info and time_info.kind == "PERIOD":
#         # kecuali kalau ada kata "ringkasan/ringkas" eksplisit atau model mendeteksi COMPARE
#         if "ringkasan" not in qlow and "ringkas" not in qlow and case_label != "COMPARE":
#             case_label = "SINGLE_PERIOD"

#     intent = {
#         "query": query_text,
#         "scope": scope,
#         "machines": machines if scope == "SUBSET" else [],
#         "granularity": final_gran,
#         "time_range": {"start": start, "end": end, "note": note},
#         "case": case_label,
#         "route_hints": route,
#         "meta": {
#             "model_time_label": model_pred["time"]["label"],
#             "model_time_conf":  model_pred["time"]["confidence"],
#             "model_case_conf":  model_pred["case"]["confidence"],
#             "model_scope_conf": model_pred["scope"]["confidence"],
#             "extractor_kind": time_info.kind if time_info else None
#         }
#     }
#     return intent




# # ---- Demo on your four examples ----
