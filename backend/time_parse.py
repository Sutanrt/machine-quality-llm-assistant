# ============================================================
# time_parser_id.py
#
# Parser waktu Bahasa Indonesia untuk query industrial telemetry.
#
# Fungsi utama:
#   - parse_time_complex(query_text, now=...)
#       → identifikasi interval waktu (bisa lebih dari satu)
#       → infer granularity (DAY / WEEK / MONTH / YEAR / HOUR)
#       → kasih route_hints buat downstream
#
#   - build_intent_complex(query_text, now=...)
#       → intent high-level siap dipakai buat jalankan pipeline
#         (scope ALL vs SUBSET, mesin mana, interval waktu, dsb)
#
# Catatan penting:
# - Ini "rule-based". Cocok buat Bahasa Indonesia pabrik/plant:
#   "minggu pertama Januari 2025", "2 hari terakhir", "bulan lalu",
#   "10–20 Januari 2025", "minggu 1 dan 2 bulan ini", dll.
#
# - Granularity di sini nggak mendukung jam/menit/shift.
#   Itu ditangani modul interpret intent lain (smart_predict).
#
# ============================================================
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import calendar
import re


# ------------------------------------------------------------
# Konstanta & kamus bahasa
# ------------------------------------------------------------

# Default "sekarang". Di production kamu bisa override via argumen fungsi.
_DEFAULT_NOW = datetime(2025, 10, 23)

ID_MONTHS: Dict[str, int] = {
    "januari": 1, "jan": 1, "jan.": 1,
    "februari": 2, "feb": 2, "feb.": 2,
    "maret": 3, "mar": 3, "mar.": 3,
    "april": 4, "apr": 4, "apr.": 4,
    "mei": 5,
    "juni": 6, "jun": 6, "jun.": 6,
    "juli": 7, "jul": 7, "jul.": 7,
    "agustus": 8, "agu": 8, "agt": 8, "agust": 8,
    "september": 9, "sep": 9, "sept": 9, "sep.": 9,
    "oktober": 10, "okt": 10, "okt.": 10,
    "november": 11, "nov": 11, "nov.": 11,
    "desember": 12, "des": 12, "des.": 12,
}

# indeks weekday python: Monday=0 ... Sunday=6
ID_DAYS: Dict[str, int] = {
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

# ordinal minggu
ORDS: Dict[str, int] = {
    "pertama": 1, "ke-1": 1, "kesatu": 1, "1": 1,
    "kedua": 2,   "ke-2": 2, "2": 2,
    "ketiga": 3,  "ke-3": 3, "3": 3,
    "keempat": 4, "ke-4": 4, "4": 4,
    "kelima": 5,  "ke-5": 5, "5": 5,
}


# ------------------------------------------------------------
# Struktur interval waktu hasil parsing
# ------------------------------------------------------------

@dataclass
class TimeInterval:
    start: datetime
    end: datetime
    granularity: str   # "HOUR" | "DAY" | "WEEK" | "MONTH" | "YEAR"
    note: str          # keterangan asal, misal "bulan ini", "minggu 1..2", dll.


# ------------------------------------------------------------
# Utility waktu dasar
# ------------------------------------------------------------

def _start_of_day(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _end_of_day(d: datetime) -> datetime:
    return d.replace(hour=23, minute=59, second=59, microsecond=0)

def _month_range(y: int, m: int) -> Tuple[datetime, datetime]:
    """
    Ambil rentang satu bulan penuh (awal jam 00:00:00 → akhir jam 23:59:59).
    """
    last = calendar.monthrange(y, m)[1]
    return (
        datetime(y, m, 1, 0, 0, 0),
        datetime(y, m, last, 23, 59, 59),
    )

def _year_range(y: int) -> Tuple[datetime, datetime]:
    """
    Ambil rentang satu tahun penuh.
    """
    return (
        datetime(y, 1, 1, 0, 0, 0),
        datetime(y, 12, 31, 23, 59, 59),
    )

def _monday_of(d: datetime) -> datetime:
    """
    Digeser ke Senin di minggu yang sama.
    """
    return _start_of_day(d - timedelta(days=d.weekday()))

def _week_range_from_monday(mon: datetime) -> Tuple[datetime, datetime]:
    """
    Rentang Senin..Minggu untuk minggu tertentu.
    """
    return mon, _end_of_day(mon + timedelta(days=6))

def _nth_week_in_month(y: int, m: int, n: int) -> Tuple[datetime, datetime]:
    """
    Ambil minggu ke-n dalam bulan tertentu:
    - minggu = blok Senin..Minggu yang overlap dengan bulan itu
    - n dibatasi supaya ga keluar range
    """
    ms, me = _month_range(y, m)
    cur = _monday_of(ms)

    weeks: List[Tuple[datetime, datetime]] = []
    while cur <= me:
        ws, we = _week_range_from_monday(cur)
        ws = max(ws, ms)
        we = min(we, me)
        if ws <= me and we >= ms:
            weeks.append((ws, we))
        cur += timedelta(days=7)
        if len(weeks) >= 6:
            break

    if not weeks:
        return ms, me

    n = max(1, min(n, len(weeks)))
    return weeks[n - 1]


def _merge_intervals(intervals: List[TimeInterval]) -> List[TimeInterval]:
    """
    Gabungkan interval yang overlap / bersebelahan
    SELAMA granularity sama.
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x.start)
    out = [intervals[0]]

    for it in intervals[1:]:
        last = out[-1]
        # gabung kalau saling menempel dan granularity sama
        if (
            it.start <= last.end + timedelta(seconds=1)
            and it.granularity == last.granularity
        ):
            last.end = max(last.end, it.end)
            last.note = f"{last.note} ∪ {it.note}"
        else:
            out.append(it)

    return out


# ------------------------------------------------------------
# Low-level regex parser untuk range tanggal Indonesia
# ------------------------------------------------------------

def _parse_day_month_year_span(text: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Tangkap pola:
        "10-20 Januari 2025"
        "10 s/d 20 Januari 2025"
    """
    pat = re.compile(
        r"\b(\d{1,2})\s*(?:-|–|sd|s/d|sampai|hingga)\s*(\d{1,2})\s+([A-Za-z\.]+)\s+(\d{4})",
        re.I,
    )
    m = pat.search(text)
    if not m:
        return None

    d1, d2 = int(m.group(1)), int(m.group(2))
    mon = ID_MONTHS.get(m.group(3).lower())
    year = int(m.group(4))

    if not mon:
        return None

    d1, d2 = sorted([d1, d2])
    s = datetime(year, mon, d1, 0, 0, 0)
    e = datetime(year, mon, d2, 23, 59, 59)
    return s, e


def _parse_day_to_day_month(text: str, default_year: int) -> Optional[Tuple[datetime, datetime]]:
    """
    Tangkap pola:
        "tanggal 1 sampai 30 januari"
        "1 hingga 5 feb"
    Tahun opsional → fallback ke default_year.
    """
    pat = re.compile(
        r"(?:tanggal\s*)?(\d{1,2})\s*(?:-|–|sd|s/d|sampai|hingga)\s*(\d{1,2})\s+([A-Za-z\.]+)(?:\s+(\d{4}))?",
        re.I,
    )
    m = pat.search(text)
    if not m:
        return None

    d1, d2 = int(m.group(1)), int(m.group(2))
    mon = ID_MONTHS.get(m.group(3).lower())
    year = int(m.group(4)) if m.group(4) else default_year

    if not mon:
        return None

    d1, d2 = sorted([d1, d2])
    s = datetime(year, mon, d1, 0, 0, 0)
    e = datetime(year, mon, d2, 23, 59, 59)
    return s, e


# ------------------------------------------------------------
# Parser utama rentang waktu natural language (tanpa jam/shift)
# ------------------------------------------------------------

def parse_time_complex(
    text: str,
    now: datetime = _DEFAULT_NOW,
) -> Dict[str, Any]:
    """
    Baca kalimat user (Bahasa Indonesia) → infer rentang waktu & granularitas.

    Contoh input:
      "10–20 Januari 2025"
      "2 hari terakhir"
      "minggu pertama Januari 2025"
      "bulan lalu"
      "hari ini"
      "minggu 1 dan 2 bulan ini"
      "tahun 2024"

    Return dict:
    {
        "intervals": [
            {"start": "...", "end": "...", "granularity": "...", "note": "..."},
            ...
        ],
        "granularity": "DAY" | "WEEK" | "MONTH" | "YEAR",   # paling halus
        "route_hints": [ ... ]  # hint extractor buat downstream
    }
    """

    t = text.lower()
    intervals: List[TimeInterval] = []

    # --- (0) kontekstual bulan/tahun dari teks ---
    month_mentioned: Optional[int] = None
    mo = re.search(
        r"(jan(uari)?|feb(ruari)?|mar(et)?|apr(il)?|mei|jun(i)?|jul(i)?|"
        r"ag(us?t?us)?|sep(t|tember)?|okt(ober)?|nov(ember)?|des(ember)?)",
        t,
    )
    if mo:
        month_mentioned = ID_MONTHS[mo.group(1)]

    year_mentioned: Optional[int] = None
    if "tahun lalu" in t:
        year_mentioned = now.year - 1
    elif "tahun ini" in t:
        year_mentioned = now.year
    else:
        yexp = re.search(r"(\d{4})", t)
        if yexp:
            year_mentioned = int(yexp.group(1))

    # --- (1) Rentang tanggal eksplisit ---
    dspan = _parse_day_month_year_span(t) or _parse_day_to_day_month(
        t, year_mentioned or now.year
    )
    if dspan:
        s, e = dspan
        # clamp kalau end jatuh "hari ini" / bulan berjalan
        if (
            e.date() == now.date()
            and (year_mentioned or now.year) == now.year
            and (month_mentioned or now.month) == now.month
        ):
            e = min(e, _end_of_day(now))
        intervals.append(TimeInterval(s, e, "DAY", "range tanggal"))

    # --- (2) Minggu ke-N (atau 1–2 / 1 dan 2) dalam satu bulan
    week_spans: List[Tuple[int, int]] = []

    # pola "minggu ke-1 ... sampai 2"
    for m in re.finditer(
        r"minggu\s+(ke-?\s*\d+|\d+|pertama|kedua|ketiga|keempat|kelima)\s*"
        r"(?:-|–|sd|s/d|sampai|hingga)\s*"
        r"(\d+|pertama|kedua|ketiga|keempat|kelima)",
        t,
    ):
        a = ORDS.get(m.group(1).replace(" ", ""), None) or ORDS.get(m.group(1), None)
        b = ORDS.get(m.group(2), None)
        if a and b:
            week_spans.append((min(a, b), max(a, b)))

    # pola "minggu 1 dan 2"
    m = re.search(
        r"minggu\s+((?:ke-?\s*\d+|\d+|pertama|kedua|ketiga|keempat|kelima)"
        r"(?:\s*(?:,|dan)\s*(?:ke-?\s*\d+|\d+|pertama|kedua|ketiga|keempat|kelima))*)",
        t,
    )
    if m:
        toks = re.split(r"\s*(?:,|dan)\s*", m.group(1))
        nums: List[int] = []
        for tok in toks:
            nums.append(
                ORDS.get(tok.replace(" ", ""), None)
                or ORDS.get(tok, None)
            )
        nums = [n for n in nums if n]
        week_spans += [(n, n) for n in nums]

    if week_spans:
        y_for_week = year_mentioned or now.year
        m_for_week = month_mentioned or (now.month if "bulan ini" in t else now.month)
        for a, b in week_spans:
            ws1, we1 = _nth_week_in_month(y_for_week, m_for_week, a)
            ws2, we2 = _nth_week_in_month(y_for_week, m_for_week, b)
            s = min(ws1, ws2)
            e = max(we1, we2)
            intervals.append(
                TimeInterval(s, e, "WEEK", f"minggu {a}..{b} {m_for_week:02d}-{y_for_week}")
            )

    # --- (3) Bulan ini / bulan lalu / "Jan 2025"
    if "bulan ini" in t:
        s, e = _month_range(now.year, now.month)
        e = min(e, _end_of_day(now))
        intervals.append(TimeInterval(s, e, "MONTH", "bulan ini"))

    if "bulan lalu" in t:
        y_tmp, mn_tmp = now.year, now.month - 1
        if mn_tmp == 0:
            y_tmp -= 1
            mn_tmp = 12
        s, e = _month_range(y_tmp, mn_tmp)
        intervals.append(TimeInterval(s, e, "MONTH", "bulan lalu"))

    # Pola "Januari 2025"
    MONTH_YR_RE = re.compile(
        r"(?P<mon>jan(?:uari)?|feb(?:ruari)?|mar(?:et)?|apr(?:il)?|mei|jun(?:i)?|jul(?:i)?|"
        r"ag(?:us?t?us)?|sep(?:t|tember)?|okt(?:ober)?|nov(?:ember)?|des(?:ember)?)\s+"
        r"(?P<year>\d{4})",
        re.I,
    )
    m = MONTH_YR_RE.search(t)
    if m:
        mon_name = m.group("mon").lower()
        yr_text = m.group("year")
        if yr_text:
            yr_val = int(yr_text)
            mon_val = ID_MONTHS[mon_name]
            s, e = _month_range(yr_val, mon_val)
            # clamp ke MTD kalau sama bulan sekarang
            if yr_val == now.year and mon_val == now.month:
                e = min(e, _end_of_day(now))
            intervals.append(
                TimeInterval(s, e, "MONTH", f"bulan {mon_val:02d}-{yr_val}")
            )

    # --- (4) Hari tertentu: "hari ini", "kemarin", "senin", ...
    # cari nama hari → "senin", "selasa", dst (ambil yg paling pertama ketemu)
    for name, wd in ID_DAYS.items():
        if re.search(rf"\b{name}\b", t):
            d = now - timedelta(days=(now.weekday() - wd) % 7)
            intervals.append(
                TimeInterval(_start_of_day(d), _end_of_day(d), "DAY", f"hari {name}")
            )
            break

    if "hari ini" in t:
        intervals.append(
            TimeInterval(_start_of_day(now), _end_of_day(now), "DAY", "hari ini")
        )

    if "kemarin" in t:
        d = now - timedelta(days=1)
        intervals.append(
            TimeInterval(_start_of_day(d), _end_of_day(d), "DAY", "kemarin")
        )

    # --- (5) Frasa "N hari/minggu/bulan terakhir"
    m = re.search(r"(\d+)\s*(hari|minggu|bulan)\s+terakhir", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)

        if unit == "hari":
            s = _start_of_day(now - timedelta(days=n - 1))
            e = _end_of_day(now)
            intervals.append(
                TimeInterval(s, e, "DAY", f"{n} hari terakhir")
            )

        elif unit == "minggu":
            s = _start_of_day(now - timedelta(days=7 * (n - 1)))
            e = _end_of_day(now)
            intervals.append(
                TimeInterval(s, e, "WEEK", f"{n} minggu terakhir")
            )

        else:  # "bulan"
            # mundur (n-1) bulan, lalu ambil awal bulan tsb → sekarang
            y_tmp, mon_tmp = now.year, now.month - (n - 1)
            while mon_tmp <= 0:
                y_tmp -= 1
                mon_tmp += 12
            s, _ = _month_range(y_tmp, mon_tmp)
            e = _end_of_day(now)
            intervals.append(
                TimeInterval(s, e, "MONTH", f"{n} bulan terakhir")
            )

    # --- (6) Tahun ini / tahun lalu / "tahun 2024"
    if "tahun ini" in t:
        s, e = _year_range(now.year)
        e = min(e, _end_of_day(now))
        intervals.append(TimeInterval(s, e, "YEAR", "tahun ini"))

    if "tahun lalu" in t:
        s, e = _year_range(now.year - 1)
        intervals.append(TimeInterval(s, e, "YEAR", "tahun lalu"))

    yx = re.search(r"tahun\s+(\d{4})", t)
    if yx:
        yy = int(yx.group(1))
        s, e = _year_range(yy)
        if yy == now.year:
            e = min(e, _end_of_day(now))
        intervals.append(TimeInterval(s, e, "YEAR", f"tahun {yy}"))

    # --- (7) fallback default:
    # kalau user gak sebut apa-apa → pakai "bulan ini" (MTD)
    if not intervals:
        s, e = _month_range(now.year, now.month)
        e = min(e, _end_of_day(now))
        intervals.append(
            TimeInterval(s, e, "MONTH", "default: bulan ini")
        )

    # --------------------------------------------------------
    # Merge semua interval yang mirip & pilih granularity terhalus
    # --------------------------------------------------------
    intervals = _merge_intervals(intervals)

    # pilih granularity "paling halus"
    # prioritas: HOUR < DAY < WEEK < MONTH < YEAR
    granularity_rank = {"HOUR": 0, "DAY": 1, "WEEK": 2, "MONTH": 3, "YEAR": 4}
    granularity_final = min(
        intervals,
        key=lambda z: granularity_rank[z.granularity],
    ).granularity

    # buat routing hint untuk downstream
    route_hints: List[str] = []
    if any(it.granularity == "WEEK" for it in intervals):
        route_hints.append("slot_week_extractor")
    if any(it.granularity == "MONTH" for it in intervals):
        route_hints.append("slot_month_extractor")
    if any(it.granularity == "YEAR" for it in intervals):
        route_hints.append("slot_year_extractor")
    if any("minggu" in it.note for it in intervals):
        route_hints.append("use_week_range_extractor")
    if len(intervals) > 1:
        route_hints.append("merge_intervals")

    return {
        "intervals": [
            {
                "start": iv.start.isoformat(),
                "end": iv.end.isoformat(),
                "granularity": iv.granularity,
                "note": iv.note,
            }
            for iv in intervals
        ],
        "granularity": granularity_final,
        "route_hints": route_hints,
    }


# ------------------------------------------------------------
# Ekstraksi mesin dan builder intent high-level
# ------------------------------------------------------------

_MACHINE_PATTERN = re.compile(
    r"\b([A-Z]{2,}\d{2,}[A-Z0-9]*|[A-Z]{3,}\d+)\b"
)

def extract_machines(text: str) -> List[str]:
    """
    Ambil ID mesin dari teks.
    Deteksi pola kapital+angka seperti XP888A, PQ667, XR101, dsb.
    Dibatasi unique, preserve urutan kemunculan.
    """
    seen: set[str] = set()
    out: List[str] = []
    for m in _MACHINE_PATTERN.finditer(text.upper()):
        tag = m.group(1)
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def build_intent_complex(
    query_text: str,
    now: datetime = _DEFAULT_NOW,
) -> Dict[str, Any]:
    """
    Wrapper yang bikin "intent" siap dipakai downstream
    tanpa harus manggil model ML apa pun.
    Dipakai buat fallback / patch mode.

    Output:
    {
      "query": "...",
      "scope": "ALL" | "SUBSET",
      "machines": [...],
      "granularity": "DAY"/"WEEK"/"MONTH"/"YEAR",
      "intervals": [...],          # hasil parse_time_complex()["intervals"]
      "route_hints": [...],
      "meta": { "machine_count": N }
    }
    """
    machines = extract_machines(query_text)

    # scope SUBSET kalau user sebut mesin spesifik & tidak bilang "semua"
    lower_q = query_text.lower()
    if "semua" in lower_q or "all" in lower_q:
        scope = "ALL"
    else:
        scope = "SUBSET" if machines else "ALL"

    time_info = parse_time_complex(query_text, now=now)

    return {
        "query": query_text,
        "scope": scope,
        "machines": machines if scope == "SUBSET" else [],
        "granularity": time_info["granularity"],
        "intervals": time_info["intervals"],
        "route_hints": time_info["route_hints"],
        "meta": {
            "machine_count": len(machines),
        },
    }


# ------------------------------------------------------------
# Local manual test
# ------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        "Tolong tampilkan mesin XP888A 10–20 Januari 2025, apakah normal?",
        "Tolong tampilkan mesin XP888A dan XR101 2 hari terakhir, apakah terlihat tidak normal?",
        "Apakah ada anomali mesin XP888A dan PQ667 hari ini?",
        "Anomali semua mesin minggu pertama dan kedua Januari 2025",
        "Anomali semua mesin minggu 1 dan 2 bulan ini",
        "Ringkasan semua mesin bulan lalu",
        "XP888A minggu 1-3 Feb 2024",
        "XP888A tgl 1-3 Feb 2024",
    ]
    for q in examples:
        print("\nQ:", q)
        print(build_intent_complex(q, now=_DEFAULT_NOW))

# ============================================================
# time_parse.py
# Indonesian natural language → TimeWindow(start, end, granularity, kind, note)
# ============================================================


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

# # =======================
# # PATCH: Time Parser (ID)
# # =======================
# import re
# from dataclasses import dataclass
# from datetime import datetime, timedelta
# from typing import List, Dict, Any, Tuple, Optional
# import calendar

# # set "hari ini" sesuai kebutuhanmu
# NOW = datetime(2025, 10, 23)

# ID_MONTHS = {
#     "januari":1,"jan":1,"jan.":1,
#     "februari":2,"feb":2,"feb.":2,
#     "maret":3,"mar":3,"mar.":3,
#     "april":4,"apr":4,"apr.":4,
#     "mei":5,
#     "juni":6,"jun":6,"jun.":6,
#     "juli":7,"jul":7,"jul.":7,
#     "agustus":8,"agu":8,"agt":8,"agust":8,
#     "september":9,"sep":9,"sept":9,"sep.":9,
#     "oktober":10,"okt":10,"okt.":10,
#     "november":11,"nov":11,"nov.":11,
#     "desember":12,"des":12,"des.":12,
# }

# ID_DAYS = {"senin":0,"selasa":1,"rabu":2,"kamis":3,"jumat":4,"jum’at":4,"sabtu":5,"minggu":6,"ahad":6}
# ORDS = {"pertama":1,"ke-1":1,"kesatu":1,"1":1,
#         "kedua":2,"ke-2":2,"2":2,
#         "ketiga":3,"ke-3":3,"3":3,
#         "keempat":4,"ke-4":4,"4":4,
#         "kelima":5,"ke-5":5,"5":5}

# @dataclass
# class TimeInterval:
#     start: datetime
#     end: datetime
#     granularity: str   # HOUR/DAY/WEEK/MONTH/YEAR
#     note: str

# def start_of_day(d): return d.replace(hour=0, minute=0, second=0, microsecond=0)
# def end_of_day(d):   return d.replace(hour=23, minute=59, second=59, microsecond=0)

# def month_range(y:int, m:int) -> Tuple[datetime, datetime]:
#     last = calendar.monthrange(y, m)[1]
#     return datetime(y,m,1,0,0,0), datetime(y,m,last,23,59,59)

# def year_range(y:int) -> Tuple[datetime, datetime]:
#     return datetime(y,1,1,0,0,0), datetime(y,12,31,23,59,59)

# def monday_of(d: datetime) -> datetime:
#     return start_of_day(d - timedelta(days=d.weekday()))

# def week_range_from_monday(mon: datetime) -> Tuple[datetime, datetime]:
#     return mon, end_of_day(mon + timedelta(days=6))

# def nth_week_in_month(y:int, m:int, n:int) -> Tuple[datetime, datetime]:
#     ms, me = month_range(y, m)
#     cur = monday_of(ms)
#     weeks = []
#     while cur <= me:
#         ws, we = week_range_from_monday(cur)
#         ws = max(ws, ms); we = min(we, me)
#         if ws <= me and we >= ms:
#             weeks.append((ws,we))
#         cur += timedelta(days=7)
#         if len(weeks) >= 6: break
#     if not weeks: return ms, me
#     n = max(1, min(n, len(weeks)))
#     return weeks[n-1]

# def merge_intervals(iv: List[TimeInterval]) -> List[TimeInterval]:
#     if not iv: return []
#     iv = sorted(iv, key=lambda x: x.start)
#     out = [iv[0]]
#     for it in iv[1:]:
#         last = out[-1]
#         if it.start <= last.end + timedelta(seconds=1) and it.granularity == last.granularity:
#             last.end = max(last.end, it.end)
#             last.note = f"{last.note} ∪ {it.note}"
#         else:
#             out.append(it)
#     return out

# # ---------- NEW robust date parsers ----------
# def _parse_day_month_year_span(text: str) -> Optional[Tuple[datetime, datetime]]:
#     """
#     Tangkap pola: '10-20 Januari 2025' atau '10 s/d 20 Januari 2025'
#     """
#     pat = re.compile(
#         r"\b(\d{1,2})\s*(?:-|–|sd|s/d|sampai|hingga)\s*(\d{1,2})\s+([A-Za-z\.]+)\s+(\d{4})",
#         re.I
#     )
#     m = pat.search(text)
#     if not m: return None
#     d1, d2 = int(m.group(1)), int(m.group(2))
#     mon = ID_MONTHS.get(m.group(3).lower(), None)
#     year = int(m.group(4))
#     if not mon: return None
#     d1, d2 = sorted([d1, d2])
#     s = datetime(year, mon, d1, 0, 0, 0)
#     e = datetime(year, mon, d2, 23, 59, 59)
#     return s, e

# def _parse_day_to_day_month(text: str, default_year: int) -> Optional[Tuple[datetime, datetime]]:
#     """
#     Tangkap pola: 'tanggal 1 sampai 30 januari' (tahun opsional → pakai default_year)
#     """
#     pat = re.compile(
#         r"(?:tanggal\s*)?(\d{1,2})\s*(?:-|–|sd|s/d|sampai|hingga)\s*(\d{1,2})\s+([A-Za-z\.]+)(?:\s+(\d{4}))?",
#         re.I
#     )
#     m = pat.search(text)
#     if not m: return None
#     d1, d2 = int(m.group(1)), int(m.group(2))
#     mon = ID_MONTHS.get(m.group(3).lower(), None)
#     year = int(m.group(4)) if m.group(4) else default_year
#     if not mon: return None
#     d1, d2 = sorted([d1, d2])
#     s = datetime(year, mon, d1, 0, 0, 0)
#     e = datetime(year, mon, d2, 23, 59, 59)
#     return s, e

# def parse_time_complex(text: str, now: datetime = NOW) -> Dict[str, Any]:
#     t = text.lower()
#     intervals: List[TimeInterval] = []
#     notes = []

#     # --- 0) Ambil konteks bulan/tahun (kalau disebut) supaya minggu 1–N tepat ---
#     month_mentioned = None
#     mo = re.search(r"(jan(uari)?|feb(ruari)?|mar(et)?|apr(il)?|mei|jun(i)?|jul(i)?|ag(us?t?us)?|sep(t|tember)?|okt(ober)?|nov(ember)?|des(ember)?)", t)
#     if mo: month_mentioned = ID_MONTHS[mo.group(1)]
#     year_mentioned = None
#     if "tahun lalu" in t:
#         year_mentioned = now.year - 1
#     elif "tahun ini" in t:
#         year_mentioned = now.year
#     else:
#         yexp = re.search(r"(\d{4})", t)
#         if yexp: year_mentioned = int(yexp.group(1))

#     # --- 1) Rentang tanggal eksplisit (10–20 Januari 2025 / 1 sampai 30 januari) ---
#     dspan = _parse_day_month_year_span(t) or _parse_day_to_day_month(t, year_mentioned or now.year)
#     if dspan:
#         s,e = dspan
#         if e.date() == now.date() and (year_mentioned or month_mentioned) == now.month:
#             e = min(e, end_of_day(now))
#         intervals.append(TimeInterval(s,e,"DAY","range tanggal"))
#         notes.append("date_span")

#     # --- 2) Minggu pertama/ke-N bulan X tahun Y (termuat juga 'minggu 1–2', 'minggu 1 dan 2') ---
#     week_spans = []
#     # a) 1–2
#     for m in re.finditer(r"minggu\s+(ke-?\s*\d+|\d+|pertama|kedua|ketiga|keempat|kelima)\s*(?:-|–|sd|s/d|sampai|hingga)\s*(\d+|pertama|kedua|ketiga|keempat|kelima)", t):
#         a = ORDS.get(m.group(1).replace(" ",""), None) or ORDS.get(m.group(1), None)
#         b = ORDS.get(m.group(2), None)
#         if a and b:
#             week_spans.append((min(a,b), max(a,b)))
#     # b) "1 dan 2"
#     m = re.search(r"minggu\s+((?:ke-?\s*\d+|\d+|pertama|kedua|ketiga|keempat|kelima)(?:\s*(?:,|dan)\s*(?:ke-?\s*\d+|\d+|pertama|kedua|ketiga|keempat|kelima))*)", t)
#     if m:
#         toks = re.split(r"\s*(?:,|dan)\s*", m.group(1))
#         nums = []
#         for tok in toks:
#             nums.append( ORDS.get(tok.replace(" ",""), None) or ORDS.get(tok, None) )
#         nums = [n for n in nums if n]
#         week_spans += [(n,n) for n in nums]

#     if week_spans:
#         y = year_mentioned or now.year
#         mth = month_mentioned or (now.month if "bulan ini" in t else now.month)
#         for a,b in week_spans:
#             ws1,we1 = nth_week_in_month(y,mth,a)
#             ws2,we2 = nth_week_in_month(y,mth,b)
#             s = min(ws1,ws2); e = max(we1,we2)
#             intervals.append(TimeInterval(s,e,"WEEK",f"minggu {a}..{b} {mth:02d}-{y}"))
#             notes.append("week_span")

#     # --- 3) Bulan ini/lalu/spesifik ---
#     if "bulan ini" in t:
#         s,e = month_range(now.year, now.month)
#         e = min(e, end_of_day(now))
#         intervals.append(TimeInterval(s,e,"MONTH","bulan ini"))
#     if "bulan lalu" in t:
#         y,mn = now.year, now.month-1
#         if mn==0: y-=1; mn=12
#         s,e = month_range(y,mn); intervals.append(TimeInterval(s,e,"MONTH","bulan lalu"))
#     # 'Januari 2025' / 'Feb 2025'
#     MONTH_YR_RE = re.compile(r"(?P<mon>jan(?:uari)?|feb(?:ruari)?|mar(?:et)?|apr(?:il)?|mei|jun(?:i)?|jul(?:i)?|ag(?:us?t?us)?|sep(?:t|tember)?|okt(?:ober)?|nov(?:ember)?|des(?:ember)?)\s+(?P<year>\d{4})",re.I
#     )
#     m = MONTH_YR_RE.search(t)
#     if m:
#         mon_name = m.group("mon").lower()
#         yr_text  = m.group("year")
#         if yr_text is not None:
#             yr = int(yr_text)
#             mon = ID_MONTHS[mon_name]
#             s, e = month_range(yr, mon)
#             if yr == now.year and mon == now.month:
#                 e = min(e, end_of_day(now))  # MTD
#             intervals.append(TimeInterval(s, e, "MONTH", f"bulan {mon:02d}-{yr}"))

#     # --- 4) Hari dalam minggu / hari ini / kemarin ---
#     for name, wd in ID_DAYS.items():
#         if re.search(rf"\b{name}\b", t):
#             d = now - timedelta(days=(now.weekday()-wd)%7)
#             intervals.append(TimeInterval(start_of_day(d), end_of_day(d), "DAY", f"hari {name}"))
#             break
#     if "hari ini" in t:
#         intervals.append(TimeInterval(start_of_day(now), end_of_day(now), "DAY", "hari ini"))
#     if "kemarin" in t:
#         d = now - timedelta(days=1)
#         intervals.append(TimeInterval(start_of_day(d), end_of_day(d), "DAY", "kemarin"))

#     # --- 5) N hari/minggu/bulan terakhir ---
#     m = re.search(r"(\d+)\s*(hari|minggu|bulan)\s+terakhir", t)
#     if m:
#         n = int(m.group(1)); unit = m.group(2)
#         if unit=="hari":
#             s = start_of_day(now - timedelta(days=n-1)); e = end_of_day(now)
#             intervals.append(TimeInterval(s,e,"DAY",f"{n} hari terakhir"))
#         elif unit=="minggu":
#             s = start_of_day(now - timedelta(days=7*(n-1))); e = end_of_day(now)
#             intervals.append(TimeInterval(s,e,"WEEK",f"{n} minggu terakhir"))
#         else:
#             # mundur (n-1) bulan sampai awal bulan tsb
#             y,mon = now.year, now.month-(n-1)
#             while mon<=0: y-=1; mon+=12
#             s,_ = month_range(y,mon); e = end_of_day(now)
#             intervals.append(TimeInterval(s,e,"MONTH",f"{n} bulan terakhir"))

#     # --- 6) Tahun ini/lalu/eksplisit ---
#     if "tahun ini" in t:
#         s,e = year_range(now.year); e = min(e,end_of_day(now))
#         intervals.append(TimeInterval(s,e,"YEAR","tahun ini"))
#     if "tahun lalu" in t:
#         s,e = year_range(now.year-1); intervals.append(TimeInterval(s,e,"YEAR","tahun lalu"))
#     yx = re.search(r"tahun\s+(\d{4})", t)
#     if yx:
#         yy = int(yx.group(1)); s,e = year_range(yy)
#         if yy==now.year: e=min(e,end_of_day(now))
#         intervals.append(TimeInterval(s,e,"YEAR",f"tahun {yy}"))

#     # --- default jika tidak ada apa-apa: bulan ini (MTD) ---
#     if not intervals:
#         s,e = month_range(now.year, now.month)
#         e = min(e, end_of_day(now))
#         intervals.append(TimeInterval(s,e,"MONTH","default: bulan ini"))

#     # merge + pilih granularity terhalus
#     intervals = merge_intervals(intervals)
#     order = {"HOUR":0,"DAY":1,"WEEK":2,"MONTH":3,"YEAR":4}
#     gran = min(intervals, key=lambda z: order[z.granularity]).granularity

#     route = []
#     if any(it.granularity=="WEEK" for it in intervals): route.append("slot_week_extractor")
#     if any(it.granularity=="MONTH" for it in intervals): route.append("slot_month_extractor")
#     if any(it.granularity=="YEAR" for it in intervals):  route.append("slot_year_extractor")
#     if any("week_span" in it.note for it in intervals):  route.append("use_week_range_extractor")
#     if len(intervals)>1: route.append("merge_intervals")

#     return {
#         "intervals": [{"start": it.start.isoformat(), "end": it.end.isoformat(),
#                        "granularity": it.granularity, "note": it.note} for it in intervals],
#         "granularity": gran,
#         "route_hints": route
#     }

# # ====== builder intent (pakai parser baru) ======
# MACHINE_PATTERN = re.compile(r"\b([A-Z]{2,}\d{2,}[A-Z0-9]*|[A-Z]{3,}\d+)\b")
# def extract_machines(text:str):
#     seen=set(); out=[]
#     for m in MACHINE_PATTERN.finditer(text.upper()):
#         x=m.group(1)
#         if x not in seen:
#             seen.add(x); out.append(x)
#     return out

# def build_intent_complex(query_text: str) -> Dict[str, Any]:
#     machines = extract_machines(query_text)
#     scope = "ALL" if ("semua" in query_text.lower() or "all" in query_text.lower()) else ("SUBSET" if machines else "ALL")
#     time_info = parse_time_complex(query_text, NOW)
#     return {
#         "query": query_text,
#         "scope": scope,
#         "machines": machines if scope=="SUBSET" else [],
#         "granularity": time_info["granularity"],
#         "intervals": time_info["intervals"],
#         "route_hints": time_info["route_hints"],
#         "meta": {"machine_count": len(machines)}
#     }

# if __name__ == "__main__":
#     examples = [
#         "Tolong tampilkan mesin XP888A 10–20 Januari 2025, apakah normal?",
#         "Tolong tampilkan mesin XP888A dan XR101 2 hari terakhir, apakah terlihat tidak normal?",
#         "Apakah ada anomali mesin XP888A dan PQ667 hari ini?",
#         "Anomali semua mesin minggu pertama dan kedua Januari 2025",
#         "Anomali semua mesin minggu 1 dan 2 bulan ini",
#         "Ringkasan semua mesin bulan lalu",
#         "XP888A minggu 1-3 Feb 2024",
#         "XP888A  tgl 1-3 Feb 2024",
#     ]
#     for q in examples:
#         print("\nQ:", q)
#         print(build_intent_complex(q))
