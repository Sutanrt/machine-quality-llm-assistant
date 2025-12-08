# ======================
# Helpers (internal use)
# ======================
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple




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
    
def _freq_from_gran(gran: str) -> str:
    """Map granularity (HOUR/DAY/...) -> pandas resample freq."""
    gran = (gran or "DAY").upper()
    return {
        "HOUR":  "1H",
        "DAY":   "1D",
        "WEEK":  "1W-MON",  # minggu mulai Senin
        "MONTH": "1MS",     # Month Start
        "YEAR":  "1Y",
        "ALL":   "1D",      # fallback
    }.get(gran, "1D")


def _agg_op(scope: str) -> str:
    """
    Pilih agregasi sesuai metrik-mu:
    - 'sum'  untuk throughput/total output
    - 'mean' untuk temperatur / quality metrics
    Sekarang default = 'mean' untuk semua.
    """
    return "mean" if scope == "SUBSET" else "mean"


def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope sederhana: cov(x,y)/var(x)."""
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < 2 or np.allclose(x.var(), 0):
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def _pct_change(a: float, b: float) -> Optional[float]:
    """Persentase perubahan dari a -> b."""
    if a is None or b is None or a == 0:
        return None
    return (b - a) / abs(a)


def _max_drawdown(series: pd.Series) -> float:
    """
    Maksimum penurunan relatif dari puncak ke lembah berikutnya.
    Contoh: -0.25 artinya turun 25% dari puncak.
    """
    if len(series) == 0:
        return 0.0
    cummax = series.cummax()
    dd = (series - cummax) / cummax.replace(0, np.nan)
    return float(dd.min()) if len(series) else 0.0


def _zscore_outliers(series: pd.Series, z: float = 3.0) -> List[pd.Timestamp]:
    """(Dipakai di versi lama) Cari indeks nilai |z| >= z."""
    if len(series) < 5 or series.std(ddof=0) == 0:
        return []
    zs = (series - series.mean()) / series.std(ddof=0)
    return list(series.index[(zs.abs() >= z)])


def _auto_freq(start, end):
    """
    Heuristik frekuensi dari lebar rentang waktu.
    - ≤3 hari      -> 1H
    - ≤35 hari     -> 1D
    - ≤200 hari    -> 1W-MON
    - lainnya      -> 1MS
    """
    if start is None or end is None:
        return "1D"
    span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    if span <= 3 * 24 * 3600:
        return "1H"
    if span <= 35 * 24 * 3600:
        return "1D"
    if span <= 200 * 24 * 3600:
        return "1W-MON"
    return "1MS"


def extract_subintent(query: str) -> str:
    """
    Klasifikasi apa yang diminta user di teks query:
    - summary
    - trend / min_only / max_only / minmax / trend_min / trend_max
    """
    q = query.lower()
    has_trend = any(k in q for k in ["tren", "trend", "fluktuasi"])
    has_min = "min" in q or "minimum" in q or "terendah" in q
    has_max = "max" in q or "maksimum" in q or "tertinggi" in q
    has_summary = "ringkasan" in q or "summary" in q

    if has_summary or (has_min and has_max and has_trend):
        return "summary"
    if has_trend and has_min and not has_max:
        return "trend_min"
    if has_trend and has_max and not has_min:
        return "trend_max"
    if has_min and has_max and not has_trend:
        return "minmax"
    if has_trend and not (has_min or has_max):
        return "trend"
    if has_min and not has_trend:
        return "min_only"
    if has_max and not has_trend:
        return "max_only"
    return "summary"


# ======================
# Core analyzer
# ======================

def analyze_trend(
    df: pd.DataFrame,
    intent: dict,
    ts_col: str = "ts",
    machine_col: str = "machine_id",
    value_col: str = "value",
    agg: Optional[str] = None,
    slope_threshold: float = 0.0001,
    recent_ratio: float = 0.3,
    subintent: str = "summary",
) -> Dict[str, Any]:
    """
    Ambil df mentah semua mesin, filter sesuai intent (time range, mesin),
    resample jadi timeseries agregat, lalu hitung:
    - tren (arah up/down/flat + %change)
    - min/max
    - volatilitas (std, cv)
    - outliers (|z|>=3)
    - max_drawdown

    Return dict hasil analisis + ringkasan teks.
    """

    # --- sanity check kolom
    assert all(
        c in df.columns for c in [ts_col, machine_col, value_col]
    ), "Kolom ts/machine_id/value wajib ada"

    # --- ambil parameter intent
    start = intent["time_range"]["start"]
    end = intent["time_range"]["end"]
    scope = intent["scope"]
    gran = (intent.get("granularity") or "DAY").upper()
    machines = intent.get("machines", []) or []

    # --- basic clean
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col, value_col])

    # filter waktu
    if start:
        df = df[df[ts_col] >= pd.Timestamp(start)]
    if end:
        df = df[df[ts_col] <= pd.Timestamp(end)]

    # filter mesin kalau SUBSET
    if scope == "SUBSET" and machines:
        df = df[df[machine_col].isin(machines)]

    # kalau kosong -> langsung keluar
    if df.empty:
        return {
            "empty": True,
            "reason": "No data in range/selection",
            "intent": intent,
        }

    # --- tentukan freq resample
    pref = _freq_from_gran(gran)  # preferensi dari intent
    auto = _auto_freq(start, end) # freq heuristik dari rentang waktu

    # pilih yang lebih halus (index lebih kecil berarti lebih halus)
    order = ["1H", "1D", "1W-MON", "1MS", "1Y"]
    freq = pref if order.index(pref) < order.index(auto) else auto

    # --- agregasi lintas mesin
    if agg is None:
        agg = "mean"  # ganti jadi 'sum' kalau metrik kamu berupa total produksi dll.

    g = (
        df[[ts_col, value_col]]
        .set_index(ts_col)
        .resample(freq)
        .agg({value_col: agg})
        .dropna()
    )

    # --- cek cukup titik buat analisis tren
    sub = (subintent or "").lower()
    needs_trend = ("trend" in sub) or (sub == "summary") or (sub == "")
    needs_min = ("min" in sub) or (sub in {"summary", "min_only", "minmax"})
    needs_max = ("max" in sub) or (sub in {"summary", "max_only", "minmax"})

    # kalau kurang titik, coba turunin freq
    if len(g) < 2 and needs_trend:
        if freq in ("1MS", "1W-MON"):
            g = g.resample("1D").mean().dropna()
            freq = "1D"
        elif freq == "1D":
            g = g.resample("1H").mean().dropna()
            freq = "1H"

    # masih kurang titik → fallback jawab min/max aja
    if len(g) < 2 and needs_trend:
        if needs_min or needs_max:
            min_val = float(g[value_col].min()) if len(g) else float("nan")
            max_val = float(g[value_col].max()) if len(g) else float("nan")
            tmin = g[value_col].idxmin().isoformat() if len(g) else None
            tmax = g[value_col].idxmax().isoformat() if len(g) else None

            return {
                "intent": intent,
                "granularity_used": freq,
                "series_points": int(len(g)),
                "summary_text": (
                    (f"Min {min_val:.3f} ({tmin[:10]}). " if needs_min and tmin else "")
                    + (f"Max {max_val:.3f} ({tmax[:10]})." if needs_max and tmax else "")
                ).strip()
                or "Tidak cukup titik untuk tren.",
                "summary": {
                    "direction": None,
                    "pct_change": None,
                    "min": {"value": min_val, "ts": tmin},
                    "max": {"value": max_val, "ts": tmax},
                    "volatility": None,
                    "outliers": {
                        "count": 0,
                        "points": [],
                    },
                    "max_drawdown": None,
                },
                "data_preview": g.tail(5).reset_index().to_dict(orient="records"),
            }

        return {
            "empty": True,
            "reason": "Too few points after resampling",
            "intent": intent,
        }

    # ======================
    #   Hitung metrik tren
    # ======================

    # slope
    x = np.arange(len(g))
    y = g[value_col].values
    slope = _linreg_slope(x, y)

    # bandingin bagian awal vs bagian akhir
    split_idx = max(1, int(len(g) * (1 - recent_ratio)))
    prev_win = g[value_col].iloc[:split_idx]
    recent_win = g[value_col].iloc[split_idx:]

    mean_prev = float(prev_win.mean()) if len(prev_win) else None
    mean_recent = float(recent_win.mean()) if len(recent_win) else None
    pct = _pct_change(mean_prev, mean_recent)

    # klasifikasi arah
    direction = "flat"
    if pct is not None:
        if pct > 0.02 or slope > slope_threshold:
            direction = "up"
        if pct < -0.02 or slope < -slope_threshold:
            direction = "down"
        if abs(pct) <= 0.01 and abs(slope) <= slope_threshold:
            direction = "flat"

    # min/max
    min_val = float(g[value_col].min())
    max_val = float(g[value_col].max())
    tmin = g[value_col].idxmin().isoformat()
    tmax = g[value_col].idxmax().isoformat()

    # volatilitas
    std_val = float(g[value_col].std(ddof=0))
    mean_val = float(g[value_col].mean())
    cv_val = float(std_val / mean_val) if mean_val != 0 else 0.0

    # outlier via z-score
    den = g[value_col].std(ddof=0)
    den = den if den and den != 0 else 1.0
    zs = (g[value_col] - g[value_col].mean()) / den
    mask_spike = zs.abs() >= 3
    n_spikes = int(mask_spike.sum())

    outlier_points = []
    if n_spikes > 0:
        tmp = zs[mask_spike].abs().sort_values(ascending=False)
        for ts in tmp.index[:10]:
            outlier_points.append(
                {
                    "ts": ts.isoformat(),
                    "value": float(g.loc[ts, value_col]),
                    "z": float(zs.loc[ts]),
                }
            )

    # drawdown
    mdd = _max_drawdown(g[value_col])

    # flag fluktuasi besar
    fluktuasi_besar = (cv_val > 0.10) or (n_spikes >= 1)

    # ======================
    #   Build summary text
    # ======================
    text_parts = []

    if needs_trend:
        dir_map = {
            "up": "meningkat",
            "down": "menurun",
            "flat": "stabil",
            None: "-",
        }
        if fluktuasi_besar:
            text_parts.append(
                f"Terdapat fluktuasi besar (CV={cv_val:.6f}, outlier={n_spikes})."
            )
        else:
            pct_txt = f"{(pct*100):.2f}%" if pct is not None else "NA"
            text_parts.append(
                f"Tren {dir_map.get(direction)} ({pct_txt}). "
                f"(CV={cv_val:.6f})."
            )

    if needs_min:
        text_parts.append(f"Min {min_val:.6f} ({tmin[:10]}).")
    if needs_max:
        text_parts.append(f"Max {max_val:.6f} ({tmax[:10]}).")

    summary_text = " ".join(text_parts).strip()

    return {
        "intent": intent,
        "series_points": int(len(g)),
        "granularity_used": freq,
        "summary_text": summary_text,
        "summary": {
            "direction": direction,
            "pct_change": pct,  # bisa None
            "min": {"value": min_val, "ts": tmin},
            "max": {"value": max_val, "ts": tmax},
            "volatility": {"std": std_val, "mean": mean_val, "cv": cv_val},
            "outliers": {
                "count": n_spikes,
                "points": outlier_points,
            },
            "max_drawdown": mdd,  # <--- DITAMBAHKAN
        },
        "data_preview": g.tail(5).reset_index().to_dict(orient="records"),
    }


# ======================
# Text formatter helpers
# ======================

def format_trend_response(result: Dict[str, Any], subintent: str) -> str:
    """
    Versi singkat (kalimat2 pendek).
    NOTE: Aman untuk pct_change=None.
    """
    if result.get("empty"):
        return f"Tidak ada data ({result['reason']})"

    s = result["summary"]
    out = []

    # trend
    if "trend" in subintent or subintent == "summary":
        pct_val = (
            f"{(s['pct_change'] * 100):.1f}%"
            if s.get("pct_change") is not None
            else "NA"
        )
        out.append(f"Tren {s['direction']} ({pct_val} perubahan).")

    # min
    if "min" in subintent or subintent in {"summary", "min_only", "minmax"}:
        out.append(
            f"Nilai minimum {s['min']['value']:.2f} "
            f"pada {s['min']['ts'][:10]}."
        )

    # max
    if "max" in subintent or subintent in {"summary", "max_only", "minmax"}:
        out.append(
            f"Nilai maksimum {s['max']['value']:.2f} "
            f"pada {s['max']['ts'][:10]}."
        )

    # volatility + drawdown
    if subintent == "summary":
        out.append(
            f"Volatilitas {s['volatility']['cv']:.2f}, "
            f"drawdown {s['max_drawdown']:.2%}."
        )

    return " ".join(out)


def pretty_response(
    result: Dict[str, Any],
    subintent: str,
    places: int = 10,
    decimal_comma: bool = True,
    show_outliers: int = 3,
) -> str:
    """
    Versi human-friendly, bisa listing outlier.
    """
    def fmt_num(x, places=10, percent=False, decimal_comma=True):
        if (
            x is None
            or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
        ):
            return "NA"
        if percent:
            x = x * 100.0
        s = f"{x:.{places}f}"
        return s.replace(".", ",") if decimal_comma else s

    if result.get("empty"):
        return f"Tidak ada data ({result.get('reason','')})"

    s = result["summary"]
    sub = (subintent or "summary").lower()

    needs_trend = ("trend" in sub) or (sub == "summary")
    needs_min = ("min" in sub) or (sub in {"summary", "min_only", "minmax"})
    needs_max = ("max" in sub) or (sub in {"summary", "max_only", "minmax"})

    parts = []

    # trend section
    if needs_trend:
        dir_map = {"up": "meningkat", "down": "menurun", "flat": "stabil", None: "-"}
        pct = s.get("pct_change")
        cv = s["volatility"]["cv"]
        outlier_count = s.get("outliers", {}).get("count", 0)
        flg = (cv > 0.10) or (outlier_count >= 1)

        if flg:
            parts.append(
                f"Terdapat fluktuasi besar "
                f"(CV={fmt_num(cv,6,False,decimal_comma)}, "
                f"outlier={outlier_count})."
            )
        else:
            parts.append(
                f"Tren {dir_map.get(s.get('direction'))} "
                f"({fmt_num(pct, places, True, decimal_comma)}%). "
                f"CV={fmt_num(cv,6,False,decimal_comma)}."
            )

    # min/max section
    if needs_min and s.get("min"):
        parts.append(
            f"Min {fmt_num(s['min']['value'], places, False, decimal_comma)} "
            f"({str(s['min']['ts'])[:10]})."
        )
    if needs_max and s.get("max"):
        parts.append(
            f"Max {fmt_num(s['max']['value'], places, False, decimal_comma)} "
            f"({str(s['max']['ts'])[:10]})."
        )

    # outlier details
    pts = s.get("outliers", {}).get("points", []) or []
    if pts:
        lines = []
        for p in pts[:show_outliers]:
            lines.append(
                f"- {p['ts'][:19]} | "
                f"nilai={fmt_num(p['value'], places, False, decimal_comma)} | "
                f"z={fmt_num(p['z'], 4, False, decimal_comma)}"
            )
        parts.append("Contoh outlier:\n" + "\n".join(lines))

    return " ".join(parts)
# =================================================================================
#CARA PAKAI
# from trend_analysis import analyze_trend, extract_subintent, pretty_response

# subintent = extract_subintent(query)
# result = analyze_trend(df_raw, intent, subintent=subintent)

# print(pretty_response(result, subintent))
# =================================================================================


# import pandas as pd
# import numpy as np
# from dataclasses import dataclass
# from typing import Optional, Dict, Any, List, Tuple
# from datetime import datetime

# # ===== helpers =====
# def _freq_from_gran(gran: str) -> str:
#     gran = (gran or "DAY").upper()
#     return {
#         "HOUR": "1H",
#         "DAY": "1D",
#         "WEEK": "1W-MON",   # minggu mulai Senin
#         "MONTH": "1MS",     # Month Start
#         "YEAR": "1Y",
#         "ALL": "1D",        # fallback
#     }.get(gran, "1D")

# def _agg_op(scope: str) -> str:
#     # Pilih agregasi sesuai metrik-mu:
#     #  - "sum" untuk output/throughput
#     #  - "mean" untuk temperatur, kualitas, dll.
#     return "mean" if scope == "SUBSET" else "mean"

# def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
#     # slope sederhana: cov(x,y)/var(x)
#     x = x.astype(float)
#     y = y.astype(float)
#     if len(x) < 2 or np.allclose(x.var(), 0):
#         return 0.0
#     return np.cov(x, y, bias=True)[0,1] / np.var(x)

# def _pct_change(a: float, b: float) -> Optional[float]:
#     # % change dari a -> b (mis. rata-rata window lama -> baru)
#     if a is None or b is None or a == 0:
#         return None
#     return (b - a) / abs(a)

# def _max_drawdown(series: pd.Series) -> float:
#     # maksimum penurunan relatif dari puncak ke lembah berikutnya
#     cummax = series.cummax()
#     dd = (series - cummax) / cummax.replace(0, np.nan)
#     return float(dd.min()) if len(series) else 0.0

# def _zscore_outliers(series: pd.Series, z=3.0) -> List[pd.Timestamp]:
#     if len(series) < 5 or series.std(ddof=0) == 0:
#         return []
#     zs = (series - series.mean())/series.std(ddof=0)
#     return list(series.index[(zs.abs() >= z)])
    
# def extract_subintent(query: str) -> str:
#     q = query.lower()
#     has_trend = any(k in q for k in ["tren", "trend", "fluktuasi"])
#     has_min = "min" in q or "minimum" in q or "terendah" in q
#     has_max = "max" in q or "maksimum" in q or "tertinggi" in q
#     has_summary = "ringkasan" in q or "summary" in q

#     if has_summary or (has_min and has_max and has_trend):
#         return "summary"
#     if has_trend and has_min and not has_max:
#         return "trend_min"
#     if has_trend and has_max and not has_min:
#         return "trend_max"
#     if has_min and has_max and not has_trend:
#         return "minmax"
#     if has_trend and not (has_min or has_max):
#         return "trend"
#     if has_min and not has_trend:
#         return "min_only"
#     if has_max and not has_trend:
#         return "max_only"
#     return "summary"

# # ===== core analyzer =====
# import pandas as pd
# import numpy as np

# def _auto_freq(start, end):
#     """Heuristik frekuensi dari lebar rentang."""
#     if start is None or end is None:
#         return "1D"
#     span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
#     # <= 3 hari → 1H ; <= 35 hari → 1D ; <= 200 hari → 1W ; lainnya → 1MS
#     if span <= 3*24*3600:      return "1H"
#     if span <= 35*24*3600:     return "1D"
#     if span <= 200*24*3600:    return "1W-MON"
#     return "1MS"

# def analyze_trend(
#     df: pd.DataFrame,
#     intent: dict,
#     ts_col="ts",
#     machine_col="machine_id",
#     value_col="value",
#     agg=None,
#     slope_threshold=0.0001,
#     recent_ratio=0.3,
#     subintent: str = "summary",
# ):
#     assert all(c in df.columns for c in [ts_col, machine_col, value_col]), "Kolom ts/machine_id/value wajib ada"

#     start = intent["time_range"]["start"]
#     end   = intent["time_range"]["end"]
#     scope = intent["scope"]
#     gran  = (intent.get("granularity") or "DAY").upper()
#     machines = intent.get("machines", []) or []

#     df = df.copy()
#     df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
#     df = df.dropna(subset=[ts_col, value_col])

#     if start: df = df[df[ts_col] >= pd.Timestamp(start)]
#     if end:   df = df[df[ts_col] <= pd.Timestamp(end)]

#     if scope == "SUBSET" and machines:
#         df = df[df[machine_col].isin(machines)]

#     if df.empty:
#         return {"empty": True, "reason": "No data in range/selection", "intent": intent}

#     # Pilih freq: gunakan granularity jika ‘masuk akal’, jika tidak auto turun ke yang lebih halus
#     pref = {"HOUR":"1H","DAY":"1D","WEEK":"1W-MON","MONTH":"1MS","YEAR":"1Y","ALL":"1D"}.get(gran, "1D")
#     auto = _auto_freq(start, end)
#     # Ambil yang lebih halus di antara keduanya
#     order = ["1H","1D","1W-MON","1MS","1Y"]
#     freq = pref if order.index(pref) < order.index(auto) else auto

#     # Agregasi antar mesin
#     if agg is None:
#         agg = "mean"   # ganti "sum" kalau metrik adalah throughput

#     g = (df[[ts_col, value_col]]
#          .set_index(ts_col)
#          .resample(freq)
#          .agg({value_col: agg})
#          .dropna())

#     # Jika user cuma minta min/max, kita tidak butuh >= 2 titik
#     sub = (subintent or "").lower()
#     needs_trend = any(k in sub for k in ["trend", "summary"]) or sub == ""  # summary mengandung trend
#     needs_min = ("min" in sub) or (sub in {"summary","min_only","minmax"})
#     needs_max = ("max" in sub) or (sub in {"summary","max_only","minmax"})

#     if len(g) < 2 and needs_trend:
#         # turunkan frekuensi lagi kalau bisa
#         if freq == "1MS":
#             g = g.resample("1D").mean().dropna()
#         elif freq == "1W-MON":
#             g = g.resample("1D").mean().dropna()
#         elif freq == "1D":
#             g = g.resample("1H").mean().dropna()

#     if len(g) < 2 and needs_trend:
#         # Masih kurang titik, tapi tetap bisa jawab min/max jika diminta
#         if needs_min or needs_max:
#             min_val = float(g[value_col].min()) if len(g) else float("nan")
#             max_val = float(g[value_col].max()) if len(g) else float("nan")
#             tmin = g[value_col].idxmin().isoformat() if len(g) else None
#             tmax = g[value_col].idxmax().isoformat() if len(g) else None
#             return {
#                 "intent": intent,
#                 "granularity_used": freq,
#                 "series_points": int(len(g)),
#                 "summary_text": (
#                     (f"Min {min_val:.3f} ({tmin[:10]}). " if needs_min and tmin else "") +
#                     (f"Max {max_val:.3f} ({tmax[:10]})."  if needs_max and tmax else "")
#                 ).strip() or "Tidak cukup titik untuk tren.",
#                 "summary": {
#                     "direction": None, "pct_change": None,
#                     "min": {"value": min_val, "ts": tmin},
#                     "max": {"value": max_val, "ts": tmax},
#                 },
#                 "data_preview": g.tail(5).reset_index().to_dict(orient="records")
#             }
#         return {"empty": True, "reason": "Too few points after resampling", "intent": intent}

#     # --- hitung metrik tren
#     x = np.arange(len(g))
#     y = g[value_col].values
#     slope = float(np.cov(x, y, bias=True)[0,1] / (np.var(x) if np.var(x) != 0 else 1))

#     split = max(1, int(len(g)*(1-0.3)))
#     prev = g[value_col].iloc[:split]; recent = g[value_col].iloc[split:]
#     mean_prev = float(prev.mean()) if len(prev) else None
#     mean_recent = float(recent.mean()) if len(recent) else None
#     pct = ( (mean_recent - mean_prev)/abs(mean_prev) ) if (mean_prev not in (None,0)) else None

#     direction = "flat"
#     if pct is not None:
#         if pct > 0.02 or slope > slope_threshold: direction = "up"
#         if pct < -0.02 or slope < -slope_threshold: direction = "down"
#         if abs(pct) <= 0.01 and abs(slope) <= slope_threshold: direction = "flat"

#     min_val = float(g[value_col].min()); max_val = float(g[value_col].max())
#     tmin = g[value_col].idxmin().isoformat(); tmax = g[value_col].idxmax().isoformat()

#     # compose teks sesuai subintent
#     parts = []
#     if needs_trend:
#         parts.append(f"Tren { {'up':'meningkat','down':'menurun','flat':'stabil'}[direction] }"
#                      + (f" ({pct*100:.1f}% change)" if pct is not None else "" ) + ".")
#     if needs_min:
#         parts.append(f"Min {min_val:.3f} ({tmin[:10]}).")
#     if needs_max:
#         parts.append(f"Max {max_val:.3f} ({tmax[:10]}).")
#     # --- tambahkan ini setelah menghitung min/max & direction ---

#     # # hitung fluktuasi (volatilitas & outlier)
#     # std = float(g[value_col].std(ddof=0))
#     # mean = float(g[value_col].mean())
#     # cv = float(std / mean) if mean != 0 else 0.0  # coefficient of variation
    
#     # # deteksi spike/outlier sederhana
#     # zscore = (g[value_col] - g[value_col].mean()) / (g[value_col].std(ddof=0) or 1)
#     # spikes = g[value_col][abs(zscore) >= 3]  # nilai ekstrem ±3σ
#     # n_spikes = len(spikes)
    
#     # # --- tentukan apakah fluktuasi besar ---
#     # fluktuasi_besar = (cv > 0.1) or (n_spikes >= 1)  # threshold bisa diatur

#         # --- Volatilitas & Outlier detail ---
#     std = float(g[value_col].std(ddof=0))
#     mean = float(g[value_col].mean())
#     cv = float(std / mean) if mean != 0 else 0.0

#     # Z-score untuk tiap titik
#     den = g[value_col].std(ddof=0)
#     den = den if den and den != 0 else 1.0
#     zs = (g[value_col] - g[value_col].mean()) / den
    
#     # Outlier = |z| >= 3
#     mask_spike = zs.abs() >= 3
#     spike_series = g.loc[mask_spike, value_col]
#     n_spikes = int(mask_spike.sum())

#     # Simpan detail (urutkan menurut |z| desc, batasi 10 titik)
#     outlier_points = []
#     if n_spikes > 0:
#         tmp = zs[mask_spike].abs().sort_values(ascending=False)
#         for ts in tmp.index[:10]:
#             outlier_points.append({
#                 "ts": ts.isoformat(),
#                 "value": float(g.loc[ts, value_col]),
#                 "z": float(zs.loc[ts])
#             })
    
#     # Flag fluktuasi besar (atur threshold sesuai kebutuhan)
#     fluktuasi_besar = (cv > 0.10) or (n_spikes >= 1)

#     # ---------- Compose text sesuai subintent ----------
#     parts = []
#     if needs_trend:
#         dir_map = {"up":"meningkat","down":"menurun","flat":"stabil", None:"-"}
#         if fluktuasi_besar:
#             parts.append(
#                 f"Terdapat fluktuasi besar (CV={cv:.6f}, outlier={n_spikes})."
#             )
#         else:
#             pct_txt = f"{(pct*100):.10f}%" if pct is not None else "NA"
#             parts.append(
#                 f"Tren {dir_map.get(direction)} ({pct_txt}). (CV={cv:.6f})."
#             )
#     if needs_min:
#         parts.append(f"Min {min_val:.10f} ({tmin[:10]}).")
#     if needs_max:
#         parts.append(f"Max {max_val:.10f} ({tmax[:10]}).")
    
#     return {
#         "intent": intent,
#         "series_points": int(len(g)),
#         "granularity_used": freq,
#         "summary_text": " ".join(parts).strip(),
#         "summary": {
#             "direction": direction,
#             "pct_change": pct,
#             "min": {"value": min_val, "ts": tmin},
#             "max": {"value": max_val, "ts": tmax},
#             "volatility": {"std": std, "mean": mean, "cv": cv},
#             "outliers": {
#                 "count": n_spikes,
#                 "points": outlier_points  # <-- detail outlier
#             },
#         },
#         "data_preview": g.tail(5).reset_index().to_dict(orient="records"),
#     }





    

#     # if needs_trend:
#     #     if fluktuasi_besar:
#     #         parts.append(f"Terdapat fluktuasi besar (CV={cv:.3f}, outlier={n_spikes}).")
#     #     else:
#     #         parts.append(
#     #             f"Tren { {'up':'meningkat','down':'menurun','flat':'stabil'}[direction] }"
#     #             + (f" ({pct*100:.10f}% change)" if pct is not None else "")
#     #             + f" (CV={cv:.3f})."
#     #         )
#     # return {
#     #     "intent": intent,
#     #     "series_points": int(len(g)),
#     #     "granularity_used": freq,
#     #     "summary_text": " ".join(parts) or "OK",
#     #     "summary" : {
#     #         "direction": direction,
#     #         "pct_change": pct,
#     #         "min": {"value": min_val, "ts": tmin},
#     #         "max": {"value": max_val, "ts": tmax},
#     #         "volatility": {"std": std, "cv": cv},
#     #         "outliers": n_spikes,
#     #     },
#     #     "data_preview": g.tail(5).reset_index().to_dict(orient="records"),
#     # }


# def format_trend_response(result, subintent: str):
#     if result.get("empty"):
#         return f"Tidak ada data ({result['reason']})"

#     s = result["summary"]
#     text_parts = []

#     if "trend" in subintent or subintent == "summary":
#         text_parts.append(f"Tren {s['direction']} ({s['pct_change']*100:.1f}% perubahan).")

#     if "min" in subintent or subintent in {"summary", "min_only", "minmax"}:
#         text_parts.append(f"Nilai minimum {s['min']['value']:.2f} pada {s['min']['ts'][:10]}.")

#     if "max" in subintent or subintent in {"summary", "max_only", "minmax"}:
#         text_parts.append(f"Nilai maksimum {s['max']['value']:.2f} pada {s['max']['ts'][:10]}.")

#     if subintent == "summary":
#         text_parts.append(f"Volatilitas {s['volatility']['cv']:.2f}, drawdown {s['max_drawdown']:.2%}.")

#     return " ".join(text_parts)


# def pretty_response(result: dict, subintent: str, places=10, decimal_comma=True, show_outliers=3):
#     import numpy as np
#     def fmt_num(x, places=10, percent=False, decimal_comma=True):
#         if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
#             return "NA"
#         if percent: x = x * 100.0
#         s = f"{x:.{places}f}"
#         return s.replace(".", ",") if decimal_comma else s

#     if result.get("empty"):
#         return f"Tidak ada data ({result.get('reason','')})"

#     s = result["summary"]
#     sub = (subintent or "summary").lower()
#     needs_trend = ("trend" in sub) or (sub == "summary")
#     needs_min   = ("min" in sub) or (sub in {"summary","min_only","minmax"})
#     needs_max   = ("max" in sub) or (sub in {"summary","max_only","minmax"})

#     parts = []
#     if needs_trend:
#         dir_map = {"up":"meningkat","down":"menurun","flat":"stabil", None:"-"}
#         pct = s.get("pct_change")
#         cv  = s["volatility"]["cv"]
#         flg = (cv > 0.10) or (s.get("outliers",{}).get("count",0) >= 1)
#         if flg:
#             parts.append(f"Terdapat fluktuasi besar (CV={fmt_num(cv,6,False,decimal_comma)}, outlier={s['outliers']['count']}).")
#         else:
#             parts.append(
#                 f"Tren {dir_map.get(s.get('direction'))} "
#                 f"({fmt_num(pct, places, True, decimal_comma)}%). "
#                 f"CV={fmt_num(cv,6,False,decimal_comma)}."
#             )

#     if needs_min and s.get("min"):
#         parts.append(f"Min {fmt_num(s['min']['value'], places, False, decimal_comma)} ({str(s['min']['ts'])[:10]}).")
#     if needs_max and s.get("max"):
#         parts.append(f"Max {fmt_num(s['max']['value'], places, False, decimal_comma)} ({str(s['max']['ts'])[:10]}).")

#     # tampilkan contoh outlier detail
#     pts = s.get("outliers",{}).get("points",[]) or []
#     if pts:
#         lines = []
#         for p in pts[:show_outliers]:
#             lines.append(
#                 f"- {p['ts'][:19]} | nilai={fmt_num(p['value'], places, False, decimal_comma)} | z={fmt_num(p['z'], 4, False, decimal_comma)}"
#             )
#         parts.append("Contoh outlier:\n" + "\n".join(lines))

#     return " ".join(parts)


