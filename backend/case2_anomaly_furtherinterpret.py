# ============================================================
# time_parse.py
# Indonesian natural language → TimeWindow(start, end, granularity, kind, note)
# ============================================================

from __future__ import annotations
import re
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple, Dict

# ---------------------------------
# Kamus & regex global
# ---------------------------------

ID_MONTHS = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12
}

DAYPART_RANGES = {
    "pagi":      (5, 11),
    "siang":     (11, 15),
    "sore":      (15, 18),
    "malam":     (18, 24),
    "dini_hari": (0, 5),
}

# note: kita izinkan "hari minggu" / "minggu hari" buat Sunday
DAYNAME_TO_IDX = {
    "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
    "jumat": 4, "jum’at": 4,
    "sabtu": 5,
    "hari minggu": 6, "minggu hari": 6,
}

MONTH_WORDS = "|".join(ID_MONTHS.keys())

# "10–20 januari 2025"
RE_DATE_RANGE_DAY = re.compile(
    rf"\b(\d{{1,2}})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{{1,2}})\s+({MONTH_WORDS})\s+(20\d{{2}})\b",
    re.I
)

# "1–20 januari dan februari 2025"
RE_DAY_RANGE_TWO_MONTHS = re.compile(
    rf"\b(\d{{1,2}})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{{1,2}})\s+"
    rf"({MONTH_WORDS})\s*(?:dan|&)\s*({MONTH_WORDS})\s+(20\d{{2}})\b",
    re.I
)

# "bulan januari dan februari 2025"
RE_MONTHS_LIST = re.compile(
    rf"\b(?:bulan\s+)?((?:{MONTH_WORDS})(?:\s*(?:,|dan|&)\s*(?:{MONTH_WORDS}))*)\s+(20\d{{2}})\b",
    re.I
)

# jam range
# contoh:
#   "pukul 08:00 - 10:30"
#   "jam 8 - 10"
RE_HHMM_RANGE = re.compile(
    r"\b(?:pukul|jam)?\s*(\d{1,2})(?::|\.)(\d{2})\s*"
    r"(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
    r"(\d{1,2})(?::|\.)(\d{2})\b",
    re.I,
)
RE_HHMM_RANGE_FLEX = re.compile(
    r"\b(?:pukul|jam)?\s*"
    r"(\d{1,2})(?::|\.)(\d{2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\s*"
    r"(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
    r"(\d{1,2})(?::|\.)(\d{2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\b",
    re.I,
)
RE_HH_RANGE = re.compile(
    r"\b(?:pukul|jam)\s*(\d{1,2})\s*"
    r"(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
    r"(\d{1,2})\b",
    re.I,
)
RE_HH_RANGE_FLEX = re.compile(
    r"\b(?:pukul|jam)\s*"
    r"(\d{1,2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\s*"
    r"(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
    r"(\d{1,2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\b",
    re.I,
)

# relatif
RE_LAST_X      = re.compile(r"\b(\d+)\s*(menit|jam|hari|minggu|bulan)\s+terakhir\b", re.I)
RE_WEEK_LAST   = re.compile(r"\bminggu\s+lalu\b", re.I)
RE_WEEK_THIS   = re.compile(r"\bminggu\s+ini\b", re.I)
RE_MONTH_THIS  = re.compile(r"\bbulan\s+ini\b", re.I)
RE_KEMARIN     = re.compile(r"\bkemarin\b", re.I)

# shift / bagian hari
RE_DAYPART     = re.compile(r"\b(pagi|siang|sore|malam|dini\s*hari)\b", re.I)

# nama hari
RE_DAYNAME     = re.compile(
    r"\b(senin|selasa|rabu|kamis|jumat|jum’at|sabtu|hari\s+minggu|minggu\s+hari)\b",
    re.I,
)

# tanggal tunggal
RE_ISO_YMD     = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
RE_DMY_SLASH   = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b")


# ---------------------------------
# helpers waktu
# ---------------------------------

def _now() -> datetime:
    """
    Central 'now'. Di production bisa kamu inject/bikin argumen,
    tapi default kita pakai datetime.now().
    """
    return datetime.now()

def _start_of_week(dt: datetime) -> datetime:
    """Senin 00:00:00 minggu tsb."""
    return (dt - timedelta(days=dt.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

def _end_of_week(dt: datetime) -> datetime:
    """Minggu 23:59:59 minggu tsb."""
    s = _start_of_week(dt)
    return s + timedelta(days=6, hours=23, minutes=59, seconds=59)

def _daypart_window(base_day: datetime, label: str) -> Tuple[datetime, datetime]:
    """
    Ambil rentang 'shift malam', 'pagi', dsb dalam 1 hari.
    """
    lb = label.replace(" ", "_").lower()
    h0, h1 = DAYPART_RANGES.get(lb, DAYPART_RANGES["malam"])
    start = base_day.replace(hour=h0, minute=0, second=0, microsecond=0)
    end = base_day.replace(
        hour=(h1 - 1 if h1 > 0 else 23), minute=59, second=59, microsecond=0
    )
    return start, end

def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


# ---------------------------------
# data class output
# ---------------------------------

@dataclass
class TimeWindow:
    start: datetime
    end: datetime
    granularity: str       # ex: "HOUR","DAY","WEEK","MONTH","SHIFT"
    kind: str              # ex: "RANGE_CLEAR","RANGE_REL","PERIOD","POINT","SHIFT"
    note: str = ""         # human readable note


# ---------------------------------
# parser utama
# ---------------------------------

def resolve_time_window(text: str, pred: dict) -> TimeWindow:
    """
    text: query asli user ("ada anomali mesin XP888A 7 hari terakhir?")
    pred: output model intent classifier (buat fallback kalau user gak jelas),
          minimal punya key:
              pred["time_granularity"]["label"]
              pred["time_complexity"]["label"]

    return: TimeWindow
    """
    tnow = _now()
    low = text.lower()

    # -------------------------------------------------
    # [A] Rentang hari eksplisit: "10–20 Januari 2025"
    # -------------------------------------------------
    m = RE_DATE_RANGE_DAY.search(low)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        mon = m.group(3).lower()
        year = int(m.group(4))
        start = datetime(year, ID_MONTHS[mon], d1, 0, 0, 0)
        end   = datetime(year, ID_MONTHS[mon], d2, 23, 59, 59)
        return TimeWindow(start, end, "DAY", "RANGE_CLEAR", "tanggal eksplisit (ID)")  # :contentReference[oaicite:0]{index=0}

    # -------------------------------------------------
    # [A2] Rentang hari lintas dua bulan:
    # "1–20 Januari dan Februari 2025"
    # -------------------------------------------------
    m = RE_DAY_RANGE_TWO_MONTHS.search(low)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        mon1, mon2 = m.group(3).lower(), m.group(4).lower()
        year = int(m.group(5))

        m1 = ID_MONTHS[mon1]
        m2 = ID_MONTHS[mon2]

        # pastikan urut
        if (year, m1) > (year, m2):
            m1, m2, d1, d2 = m2, m1, d2, d1

        start = datetime(year, m1, d1, 0, 0, 0)
        end   = datetime(
            year,
            m2,
            d2,
            23, 59, 59,
        )
        note = f"{d1}–{d2} {mon1}–{mon2} {year}"
        return TimeWindow(start, end, "DAY", "RANGE_CLEAR", note)  # :contentReference[oaicite:1]{index=1}

    # -------------------------------------------------
    # [B] Rentang jam eksplisit (default: hari ini)
    # contoh: "jam 8 sampai 10", "08:00 - 10:30"
    # -------------------------------------------------
    m = RE_HHMM_RANGE.search(low) or RE_HH_RANGE.search(low)
    if m:
        if len(m.groups()) == 4:
            # HH:MM - HH:MM
            h1, m1, h2, m2 = map(int, m.groups())
            start = tnow.replace(hour=h1 % 24, minute=m1 % 60, second=0, microsecond=0)
            end   = tnow.replace(hour=h2 % 24, minute=m2 % 60, second=59, microsecond=0)
        else:
            # jam H1 - H2
            h1, h2 = map(int, m.groups())
            start = tnow.replace(hour=h1 % 24, minute=0, second=0, microsecond=0)
            end   = tnow.replace(hour=h2 % 24, minute=59, second=59, microsecond=0)
        return TimeWindow(start, end, "HOUR", "RANGE_CLEAR", "jam eksplisit (hari ini)")  # :contentReference[oaicite:2]{index=2}

    # -------------------------------------------------
    # [C] Frasa relatif:
    # "30 menit terakhir", "7 hari terakhir", "2 minggu terakhir", "3 bulan terakhir"
    # -------------------------------------------------
    m = RE_LAST_X.search(low)
    if m:
        qty, unit = int(m.group(1)), m.group(2).lower()
        end = tnow

        if unit.startswith("menit"):
            start = end - timedelta(minutes=qty)
            gran = "HOUR"
        elif unit.startswith("jam"):
            start = end - timedelta(hours=qty)
            gran = "HOUR"
        elif unit.startswith("hari"):
            start = end - timedelta(days=qty)
            gran = "DAY"
        elif unit.startswith("mingg"):
            start = end - timedelta(weeks=qty)
            gran = "WEEK"
        else:
            # bulan terakhir pakai relativedelta
            start = end - relativedelta(months=qty)
            gran = "MONTH"

        note = f"{qty} {unit} terakhir"
        return TimeWindow(start, end, gran, "RANGE_REL", note)  # :contentReference[oaicite:3]{index=3}

    # -------------------------------------------------
    # [D] Frasa campuran hari-pekan + nama hari:
    # "senin minggu lalu jam 8 pagi sampai 10 pagi"
    # "rabu minggu ini shift malam"
    # -------------------------------------------------
    dn = RE_DAYNAME.search(low)
    is_week_phrase = bool(RE_WEEK_LAST.search(low) or RE_WEEK_THIS.search(low) or "minggu ini" in low)
    if dn and is_week_phrase:
        dn_str = dn.group(1).lower()

        # referensi: minggu lalu vs minggu ini
        ref = tnow - timedelta(weeks=1) if RE_WEEK_LAST.search(low) else tnow
        week_start = _start_of_week(ref)

        want_idx = DAYNAME_TO_IDX.get(dn_str)
        if want_idx is not None:
            target_day = week_start + timedelta(days=want_idx)

            # default full day
            start = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
            end   = target_day.replace(hour=23, minute=59, second=59, microsecond=0)

            # daypart? ("pagi", "malam", dll)
            dp = RE_DAYPART.search(low)
            if dp:
                start, end = _daypart_window(target_day, dp.group(1))

            # jam range spesifik? (08:00 - 10:30 dst)
            m = RE_HHMM_RANGE_FLEX.search(low) or RE_HH_RANGE_FLEX.search(low)
            if m:
                if len(m.groups()) == 4:
                    h1, mm1, h2, mm2 = map(int, m.groups())
                    start = target_day.replace(hour=h1 % 24, minute=mm1 % 60, second=0, microsecond=0)
                    end   = target_day.replace(hour=h2 % 24, minute=mm2 % 60, second=59, microsecond=0)
                else:
                    h1, h2 = map(int, m.groups())
                    start = target_day.replace(hour=h1 % 24, minute=0, second=0, microsecond=0)
                    end   = target_day.replace(hour=h2 % 24, minute=59, second=59, microsecond=0)

            note = f"{dn_str} ({'minggu lalu' if RE_WEEK_LAST.search(low) else 'minggu ini'})"
            return TimeWindow(start, end, "DAY", "RANGE_CLEAR", note)  # :contentReference[oaicite:4]{index=4}

    # -------------------------------------------------
    # [E] Daypart tanpa hari eksplisit:
    # "shift malam kemarin", "pagi tadi", "tadi malam"
    # -------------------------------------------------
    dp = RE_DAYPART.search(low)
    if dp and not (RE_WEEK_LAST.search(low) or RE_WEEK_THIS.search(low) or dn):
        base = (tnow - timedelta(days=1)) if RE_KEMARIN.search(low) else tnow
        base = base.replace(hour=0, minute=0, second=0, microsecond=0)
        start, end = _daypart_window(base, dp.group(1))

        return TimeWindow(start, end, "SHIFT", "SHIFT", f"shift {dp.group(1).lower()}")  # :contentReference[oaicite:5]{index=5}

    # -------------------------------------------------
    # [F] Satu tanggal spesifik → full day
    # "2025-01-15"
    # "15/01/2025"
    # -------------------------------------------------
    m = RE_ISO_YMD.search(low)
    if m:
        y, mo, d = map(int, m.groups())
        start = datetime(y, mo, d, 0, 0, 0)
        end   = datetime(y, mo, d, 23, 59, 59)
        return TimeWindow(start, end, "DAY", "POINT", "tanggal ISO (full day)")  # :contentReference[oaicite:6]{index=6}

    m = RE_DMY_SLASH.search(low)
    if m:
        d, mo, y = map(int, m.groups())
        start = datetime(y, mo, d, 0, 0, 0)
        end   = datetime(y, mo, d, 23, 59, 59)
        return TimeWindow(start, end, "DAY", "POINT", "tanggal dd/mm/yyyy (full day)")  # :contentReference[oaicite:7]{index=7}

    # -------------------------------------------------
    # [G] Bulan eksplisit (mungkin beberapa bulan) + tahun
    # "bulan januari dan februari 2025"
    # "januari, februari dan maret 2025"
    # -------------------------------------------------
    m = RE_MONTHS_LIST.search(low)
    if m:
        months_str, year = m.group(1), int(m.group(2))

        # pecah "januari, februari dan maret"
        names = re.split(r"\s*(?:,|dan|&)\s*", months_str, flags=re.I)
        months = sorted(ID_MONTHS[name.lower()] for name in names if name)

        if months:
            m1 = months[0]
            m2 = months[-1]

            start = datetime(year, m1, 1, 0, 0, 0)
            end   = datetime(
                year,
                m2,
                _last_day_of_month(year, m2),
                23, 59, 59,
            )

            label = (
                f"{names[0]}–{names[-1]} {year}"
                if len(months) > 1
                else f"{names[0]} {year}"
            )
            return TimeWindow(start, end, "MONTH", "PERIOD", label)  # :contentReference[oaicite:8]{index=8}

    # -------------------------------------------------
    # [H] Period generik: "minggu ini", "minggu lalu", "bulan ini"
    # -------------------------------------------------
    if RE_WEEK_THIS.search(low):
        return TimeWindow(
            _start_of_week(tnow),
            tnow,
            "WEEK",
            "PERIOD",
            "minggu ini",
        )  # :contentReference[oaicite:9]{index=9}

    if RE_WEEK_LAST.search(low):
        ref = tnow - timedelta(weeks=1)
        return TimeWindow(
            _start_of_week(ref),
            _end_of_week(ref),
            "WEEK",
            "PERIOD",
            "minggu lalu",
        )  # :contentReference[oaicite:10]{index=10}

    if RE_MONTH_THIS.search(low):
        start = tnow.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return TimeWindow(
            start,
            tnow,
            "MONTH",
            "PERIOD",
            "bulan ini",
        )  # :contentReference[oaicite:11]{index=11}

    # -------------------------------------------------
    # [I] Fallback ke prediksi model (granularity & complexity)
    # pred["time_granularity"]["label"] ex: "DAY","WEEK","MONTH"
    # pred["time_complexity"]["label"]  ex: "PERIOD","RANGE_REL"
    # -------------------------------------------------
    gran = (pred.get("time_granularity", {}) or {}).get("label", "DAY").upper()
    cplx = (pred.get("time_complexity", {}) or {}).get("label", "PERIOD").upper()

    if gran == "MONTH" and cplx in {"PERIOD", "RANGE_REL"}:
        start = tnow.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return TimeWindow(start, tnow, "MONTH", cplx, "fallback model → bulan berjalan")  # :contentReference[oaicite:12]{index=12}

    if gran == "WEEK" and cplx in {"PERIOD", "RANGE_REL"}:
        return TimeWindow(
            _start_of_week(tnow),
            tnow,
            "WEEK",
            cplx,
            "fallback model → minggu berjalan",
        )  # :contentReference[oaicite:13]{index=13}

    if gran == "DAY" and cplx in {"PERIOD", "RANGE_REL"}:
        start = tnow.replace(hour=0, minute=0, second=0, microsecond=0)
        return TimeWindow(
            start,
            tnow,
            "DAY",
            cplx,
            "fallback model → hari ini",
        )  # :contentReference[oaicite:14]{index=14}

    # -------------------------------------------------
    # [J] Ultimate fallback:
    # kalau semua gagal, pakai "24 jam terakhir"
    # -------------------------------------------------
    return TimeWindow(
        tnow - timedelta(days=1),
        tnow,
        "DAY",
        "RANGE_REL",
        "fallback 24 jam terakhir",
    )  # :contentReference[oaicite:15]{index=15}

# from __future__ import annotations
# import re
# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# from dataclasses import dataclass
# import calendar

# MONTH_WORDS = "|".join([
#     "januari","februari","maret","april","mei","juni",
#     "juli","agustus","september","oktober","november","desember"
# ])
# RE_MONTHS_LIST = re.compile(
#     rf"\b(?:bulan\s+)?((?:{MONTH_WORDS})(?:\s*(?:,|dan|&)\s*(?:{MONTH_WORDS}))*)\s+(20\d{{2}})\b",
#     re.I
# )

# RE_DAY_RANGE_TWO_MONTHS = re.compile(
#     rf"\b(\d{{1,2}})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{{1,2}})\s+"
#     rf"({MONTH_WORDS})\s*(?:dan|&)\s*({MONTH_WORDS})\s+(20\d{{2}})\b",
#     re.I
# )
# def _last_day_of_month(year:int, month:int) -> int:
#     return calendar.monthrange(year, month)[1]

# ID_MONTHS = {
#     "januari":1, "februari":2, "maret":3, "april":4, "mei":5, "juni":6,
#     "juli":7, "agustus":8, "september":9, "oktober":10, "november":11, "desember":12
# }
# DAYPART_RANGES = {"pagi":(5,11),"siang":(11,15),"sore":(15,18),"malam":(18,24),"dini_hari":(0,5)}
# DAYNAME_TO_IDX = {"senin":0,"selasa":1,"rabu":2,"kamis":3,"jumat":4,"jum’at":4,"sabtu":5,"hari minggu":6,"minggu hari":6}

# RE_DATE_RANGE_DAY = re.compile(
#     rf"\b(\d{{1,2}})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{{1,2}})\s+({'|'.join(ID_MONTHS.keys())})\s+(20\d{{2}})\b",
#     re.I
# )
# RE_HHMM_RANGE = re.compile(r"\b(?:pukul|jam)?\s*(\d{1,2})(?::|\.)(\d{2})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{1,2})(?::|\.)(\d{2})\b", re.I)
# RE_HHMM_RANGE_FLEX = re.compile(
#     r"\b(?:pukul|jam)?\s*"
#     r"(\d{1,2})(?::|\.)(\d{2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\s*"
#     r"(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
#     r"(\d{1,2})(?::|\.)(\d{2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\b",
#     re.I
# )
# RE_HH_RANGE   = re.compile(r"\b(?:pukul|jam)\s*(\d{1,2})\s*(?:-|–|—|s/d|sd|sampai|hingga|to)\s*(\d{1,2})\b", re.I)
# RE_HH_RANGE_FLEX = re.compile(
#     r"\b(?:pukul|jam)\s*"
#     r"(\d{1,2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\s*"
#     r"(?:-|–|—|s/d|sd|sampai|hingga|to)\s*"
#     r"(\d{1,2})\s*(?:pagi|siang|sore|malam|dini\s*hari)?\b",
#     re.I
# )
# RE_LAST_X     = re.compile(r"\b(\d+)\s*(menit|jam|hari|minggu|bulan)\s+terakhir\b", re.I)
# RE_WEEK_LAST  = re.compile(r"\bminggu\s+lalu\b", re.I)
# RE_WEEK_THIS  = re.compile(r"\bminggu\s+ini\b", re.I)
# RE_MONTH_THIS = re.compile(r"\bbulan\s+ini\b", re.I)
# RE_DAYPART    = re.compile(r"\b(pagi|siang|sore|malam|dini\s*hari)\b", re.I)
# RE_KEMARIN    = re.compile(r"\bkemarin\b", re.I)
# # RE_DAYNAME    = re.compile(r"\b(senin|selasa|rabu|kamis|jumat|jum’at|sabtu|minggu)\b", re.I)
# RE_DAYNAME = re.compile(r"\b(senin|selasa|rabu|kamis|jumat|jum’at|sabtu|hari\s+minggu|minggu\s+hari)\b", re.I)
# RE_ISO_YMD    = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
# RE_DMY_SLASH  = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b")




# # Nama hari TANPA 'minggu' biasa, tetapi IZINKAN 'hari minggu'


# # Rentang jam fleksibel: izinkan daypart di sisi kiri/kanan angka


# # (opsional) versi HH:MM juga fleksibel terhadap daypart sesudah menit



# def _now(): return datetime.now()
# def _start_of_week(dt): return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
# def _end_of_week(dt): s=_start_of_week(dt); return s + timedelta(days=6, hours=23, minutes=59, seconds=59)

# def _daypart_window(base_day: datetime, label: str):
#     lb = label.replace(" ", "_").lower()
#     h0,h1 = DAYPART_RANGES.get(lb, DAYPART_RANGES["malam"])
#     start = base_day.replace(hour=h0, minute=0, second=0, microsecond=0)
#     end   = base_day.replace(hour=(h1-1 if h1>0 else 23), minute=59, second=59, microsecond=0)
#     return start, end

# @dataclass
# class TimeWindow:
#     start: datetime
#     end: datetime
#     granularity: str
#     kind: str
#     note: str = ""

# def resolve_time_window(text: str, pred: dict) -> TimeWindow:
#     tnow = _now()
#     low = text.lower()

#     # === [A] Rentang hari eksplisit: 10–20 Januari 2025 ===
#     m = RE_DATE_RANGE_DAY.search(low)
#     if m:
#         d1, d2, mon, year = int(m.group(1)), int(m.group(2)), m.group(3).lower(), int(m.group(4))
#         start = datetime(year, ID_MONTHS[mon], d1, 0, 0, 0)
#         end   = datetime(year, ID_MONTHS[mon], d2, 23, 59, 59)
#         return TimeWindow(start, end, "DAY", "RANGE_CLEAR", "tanggal eksplisit (ID)")

#     # === [A2] Rentang hari lintas dua bulan: 1–20 Januari dan Februari 2025 ===
#     m = RE_DAY_RANGE_TWO_MONTHS.search(low)
#     if m:
#         d1, d2 = int(m.group(1)), int(m.group(2))
#         mon1, mon2 = m.group(3).lower(), m.group(4).lower()
#         year = int(m.group(5))
#         m1, m2 = ID_MONTHS[mon1], ID_MONTHS[mon2]
#         if (year, m1) > (year, m2):  # pastikan urut
#             m1, m2, d1, d2 = m2, m1, d2, d1
#         start = datetime(year, m1, d1, 0, 0, 0)
#         end   = datetime(year, m2, d2, 23, 59, 59)
#         return TimeWindow(start, end, "DAY", "RANGE_CLEAR", f"{d1}–{d2} {mon1}–{mon2} {year}")

#     # === [B] Rentang jam eksplisit (default: hari ini) ===
#     m = RE_HHMM_RANGE.search(low) or RE_HH_RANGE.search(low)
#     if m:
#         if len(m.groups()) == 4:
#             h1, m1, h2, m2 = map(int, m.groups())
#             start = tnow.replace(hour=h1 % 24, minute=m1 % 60, second=0, microsecond=0)
#             end   = tnow.replace(hour=h2 % 24, minute=m2 % 60, second=59, microsecond=0)
#         else:
#             h1, h2 = map(int, m.groups())
#             start = tnow.replace(hour=h1 % 24, minute=0, second=0, microsecond=0)
#             end   = tnow.replace(hour=h2 % 24, minute=59, second=59, microsecond=0)
#         return TimeWindow(start, end, "HOUR", "RANGE_CLEAR", "jam eksplisit (hari ini)")

#     # === [C] Frasa relatif: X menit/jam/hari/minggu/bulan terakhir ===
#     m = RE_LAST_X.search(low)
#     if m:
#         qty, unit = int(m.group(1)), m.group(2).lower()
#         end = tnow
#         if unit.startswith("menit"): start, gran = end - timedelta(minutes=qty), "HOUR"
#         elif unit.startswith("jam"):  start, gran = end - timedelta(hours=qty),   "HOUR"
#         elif unit.startswith("hari"): start, gran = end - timedelta(days=qty),    "DAY"
#         elif unit.startswith("mingg"):start, gran = end - timedelta(weeks=qty),   "WEEK"
#         else:                         start, gran = end - relativedelta(months=qty), "MONTH"
#         return TimeWindow(start, end, gran, "RANGE_REL", f"{qty} {unit} terakhir")

#     # === [D] Kombinasi hari-pekan + daypart/jam (prioritas tinggi) ===
#     dn = RE_DAYNAME.search(low)
#     is_week_phrase = RE_WEEK_LAST.search(low) or RE_WEEK_THIS.search(low) or "minggu ini" in low
#     if dn and is_week_phrase:
#         dn_str = dn.group(1).lower()
#         ref = (tnow - timedelta(weeks=1)) if RE_WEEK_LAST.search(low) else tnow
#         week_start = _start_of_week(ref)
#         want_idx = DAYNAME_TO_IDX.get(dn_str, None)
#         if want_idx is not None:
#             target = week_start + timedelta(days=want_idx)
#             start = target.replace(hour=0, minute=0, second=0, microsecond=0)
#             end   = target.replace(hour=23, minute=59, second=59, microsecond=0)
#             dp = RE_DAYPART.search(low)
#             if dp: start, end = _daypart_window(target, dp.group(1))
#             m = RE_HHMM_RANGE_FLEX.search(low) or RE_HH_RANGE_FLEX.search(low)
#             if m:
#                 if len(m.groups()) == 4:
#                     h1, m1, h2, m2 = map(int, m.groups())
#                     start = target.replace(hour=h1 % 24, minute=m1 % 60, second=0, microsecond=0)
#                     end   = target.replace(hour=h2 % 24, minute=m2 % 60, second=59, microsecond=0)
#                 else:
#                     h1, h2 = map(int, m.groups())
#                     start = target.replace(hour=h1 % 24, minute=0, second=0, microsecond=0)
#                     end   = target.replace(hour=h2 % 24, minute=59, second=59, microsecond=0)
#             return TimeWindow(start, end, "DAY", "RANGE_CLEAR",
#                               f"{dn_str} ({'minggu lalu' if RE_WEEK_LAST.search(low) else 'minggu ini'})")

#     # === [E] Daypart (tanpa nama-hari) → hari ini/kemarin ===
#     dp = RE_DAYPART.search(low)
#     if dp and not (RE_WEEK_LAST.search(low) or RE_WEEK_THIS.search(low) or dn):
#         base = (tnow - timedelta(days=1)) if RE_KEMARIN.search(low) else tnow
#         base = base.replace(hour=0, minute=0, second=0, microsecond=0)
#         start, end = _daypart_window(base, dp.group(1))
#         return TimeWindow(start, end, "SHIFT", "SHIFT", f"shift {dp.group(1).lower()}")

#     # === [F] Satu tanggal spesifik → sehari penuh ===
#     m = RE_ISO_YMD.search(low)
#     if m:
#         y, mo, d = map(int, m.groups())
#         return TimeWindow(datetime(y, mo, d, 0, 0, 0), datetime(y, mo, d, 23, 59, 59),
#                           "DAY", "POINT", "tanggal ISO (full day)")
#     m = RE_DMY_SLASH.search(low)
#     if m:
#         d, mo, y = map(int, m.groups())
#         return TimeWindow(datetime(y, mo, d, 0, 0, 0), datetime(y, mo, d, 23, 59, 59),
#                           "DAY", "POINT", "tanggal dd/mm/yyyy (full day)")

#     # === [G] Bulan eksplisit (satu/lebih) + tahun → bulan penuh ===
#     m = RE_MONTHS_LIST.search(low)
#     if m:
#         months_str, year = m.group(1), int(m.group(2))
#         # pecah "januari, februari dan maret"
#         names = re.split(r"\s*(?:,|dan|&)\s*", months_str, flags=re.I)
#         months = sorted(ID_MONTHS[n.lower()] for n in names if n)
#         if months:
#             m1, m2 = months[0], months[-1]
#             start = datetime(year, m1, 1, 0, 0, 0)
#             end   = datetime(year, m2, _last_day_of_month(year, m2), 23, 59, 59)
#             label = f"{names[0]}–{names[-1]} {year}" if len(months) > 1 else f"{names[0]} {year}"
#             return TimeWindow(start, end, "MONTH", "PERIOD", label)

#     # === [H] Period generik: minggu/bulan ini ===
#     if RE_WEEK_THIS.search(low):
#         return TimeWindow(_start_of_week(tnow), tnow, "WEEK", "PERIOD", "minggu ini")
#     if RE_WEEK_LAST.search(low):
#         ref = tnow - timedelta(weeks=1)
#         return TimeWindow(_start_of_week(ref), _end_of_week(ref), "WEEK", "PERIOD", "minggu lalu")
#     if RE_MONTH_THIS.search(low):
#         start = tnow.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#         return TimeWindow(start, tnow, "MONTH", "PERIOD", "bulan ini")

#     # === [I] Fallback ke label model ===
#     gran = (pred.get("time_granularity", {}) or {}).get("label", "DAY").upper()
#     cplx = (pred.get("time_complexity", {}) or {}).get("label", "PERIOD").upper()
#     if gran == "MONTH" and cplx in {"PERIOD", "RANGE_REL"}:
#         return TimeWindow(tnow.replace(day=1, hour=0, minute=0, second=0, microsecond=0), tnow,
#                           "MONTH", cplx, "fallback model → bulan berjalan")
#     if gran == "WEEK" and cplx in {"PERIOD", "RANGE_REL"}:
#         return TimeWindow(_start_of_week(tnow), tnow, "WEEK", cplx, "fallback model → minggu berjalan")
#     if gran == "DAY" and cplx in {"PERIOD", "RANGE_REL"}:
#         return TimeWindow(tnow.replace(hour=0, minute=0, second=0, microsecond=0), tnow,
#                           "DAY", cplx, "fallback model → hari ini")

#     # === [J] Terakhir: 24 jam terakhir ===
#     return TimeWindow(tnow - timedelta(days=1), tnow, "DAY", "RANGE_REL", "fallback 24 jam terakhir")



#     # --- (5) Daypart umum (tanpa nama-hari/minggu) → hari ini / kemarin
#     dp = RE_DAYPART.search(low)
#     if dp and not (RE_WEEK_LAST.search(low) or RE_WEEK_THIS.search(low) or dn):
#         base = (tnow - timedelta(days=1)) if RE_KEMARIN.search(low) else tnow
#         base = base.replace(hour=0, minute=0, second=0, microsecond=0)
#         start,end = _daypart_window(base, dp.group(1))
#         return TimeWindow(start,end,"SHIFT","SHIFT",f"shift {dp.group(1).lower()}")

#     # --- (6) Satu tanggal spesifik → full day
#     m = RE_ISO_YMD.search(low)
#     if m:
#         y,mo,d = map(int, m.groups())
#         start = datetime(y, mo, d, 0,0,0); end = datetime(y, mo, d, 23,59,59)
#         return TimeWindow(start,end,"DAY","POINT","tanggal ISO (full day)")
#     m = RE_DMY_SLASH.search(low)
#     if m:
#         d,mo,y = map(int, m.groups())
#         start = datetime(y, mo, d, 0,0,0); end = datetime(y, mo, d, 23,59,59)
#         return TimeWindow(start,end,"DAY","POINT","tanggal dd/mm/yyyy (full day)")

#     # --- (7) Period generik: minggu/bulan ini (fallback ringan)
#     if RE_WEEK_THIS.search(low):
#         return TimeWindow(_start_of_week(tnow), tnow, "WEEK", "PERIOD", "minggu ini")
#     if RE_WEEK_LAST.search(low):
#         ref = tnow - timedelta(weeks=1)
#         return TimeWindow(_start_of_week(ref), _end_of_week(ref), "WEEK", "PERIOD", "minggu lalu")
#     if RE_MONTH_THIS.search(low):
#         start = tnow.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#         return TimeWindow(start, tnow, "MONTH", "PERIOD", "bulan ini")

#     # --- (8) Fallback ke label model
#     gran = (pred.get("time_granularity",{}) or {}).get("label","DAY").upper()
#     cplx = (pred.get("time_complexity",{}) or {}).get("label","PERIOD").upper()
#     if gran=="MONTH" and cplx in {"PERIOD","RANGE_REL"}:
#         return TimeWindow(tnow.replace(day=1,hour=0,minute=0,second=0,microsecond=0), tnow, "MONTH", cplx, "fallback model → bulan berjalan")
#     if gran=="WEEK" and cplx in {"PERIOD","RANGE_REL"}:
#         return TimeWindow(_start_of_week(tnow), tnow, "WEEK", cplx, "fallback model → minggu berjalan")
#     if gran=="DAY" and cplx in {"PERIOD","RANGE_REL"}:
#         return TimeWindow(tnow.replace(hour=0,minute=0,second=0,microsecond=0), tnow, "DAY", cplx, "fallback model → hari ini")

#     # --- (9) Terakhir: 24 jam terakhir
#     return TimeWindow(tnow - timedelta(days=1), tnow, "DAY", "RANGE_REL", "fallback 24 jam terakhir")
