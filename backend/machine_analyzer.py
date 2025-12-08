import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# =========================
# Utilities
# =========================

def _auto_freq(start, end) -> str:
    """
    Pilih freq resample berdasar lebar rentang waktu.
    """
    if start is None or end is None:
        return "1D"
    span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    if span <= 3 * 24 * 3600:
        return "1H"       # ≤ 3 hari  -> hourly
    if span <= 35 * 24 * 3600:
        return "1D"       # ≤ ~5 minggu -> daily
    if span <= 200 * 24 * 3600:
        return "1W-MON"   # ≤ ~6.5 bulan -> weekly (mulai Senin)
    return "1MS"          # sisanya -> monthly (month start)


def _trend_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Hitung slope, %perubahan window terakhir vs awal, dan arah 'up/down/flat'.
    """
    if len(series) < 2:
        return {"slope": 0.0, "pct": None, "direction": None}

    y = series.values.astype(float)
    x = np.arange(len(series), dtype=float)

    varx = np.var(x)
    slope = float(np.cov(x, y, bias=True)[0, 1] / (varx if varx != 0 else 1))

    split = max(1, int(len(series) * 0.7))
    prev = series.iloc[:split]
    recent = series.iloc[split:]

    prev_mean = prev.mean()
    recent_mean = recent.mean()

    pct = None
    if np.isfinite(prev_mean) and prev_mean not in (0, 0.0) and np.isfinite(recent_mean):
        pct = (recent_mean - prev_mean) / abs(prev_mean)

    direction = "flat"
    if pct is not None:
        if pct > 0.02 or slope > 1e-4:
            direction = "up"
        if pct < -0.02 or slope < -1e-4:
            direction = "down"
        if abs(pct) <= 0.01 and abs(slope) <= 1e-4:
            direction = "flat"

    return {
        "slope": float(slope),
        "pct": None if pct is None else float(pct),
        "direction": direction,
    }


def _mad(x: pd.Series) -> (float, float):
    """
    Median Absolute Deviation (robust), buat deteksi outlier robust.
    Return (median, mad_norm).
    """
    med = x.median()
    mad = (x - med).abs().median()
    return med, (mad if mad and mad != 0 else 1.0)


def _rolling_z(series: pd.Series, win: int = 12) -> pd.Series:
    """
    Rolling z-score (mean/std tiap window).
    Cocok buat spike lokal/periodik.
    """
    if len(series) < max(5, win):
        return pd.Series(index=series.index, dtype=float)

    m = series.rolling(win, min_periods=max(3, win // 3)).mean()
    s = series.rolling(win, min_periods=max(3, win // 3)).std(ddof=0).replace(0, np.nan)
    return (series - m) / s


def _top_fluctuation_windows(series: pd.Series, win: int, k: int = 3) -> List[Dict[str, Any]]:
    """
    Cari periode dengan koefisien variasi (CV = std/mean) tertinggi.
    Kita pakai data RESAMPLED (bukan raw) biar gak terlalu noisy per-detik.
    """
    if len(series) < max(5, win):
        return []

    roll_std = series.rolling(win, min_periods=max(3, win // 3)).std(ddof=0)
    roll_mean = series.rolling(win, min_periods=max(3, win // 3)).mean().replace(0, np.nan)
    roll_cv = (roll_std / roll_mean).replace([np.inf, -np.inf], np.nan)

    sc = roll_cv.fillna(0)
    idxs = sc.nlargest(k).index

    out = []
    for ts in idxs:
        # ambil window terakhir sepanjang `win` titik sampai ts
        try:
            seg = series.loc[:ts].iloc[-win:]
        except Exception:
            continue
        if seg.empty:
            continue

        seg_std = float(seg.std(ddof=0))
        seg_mean = float(seg.mean())
        seg_cv = float(seg_std / seg_mean) if seg_mean != 0 else float("nan")

        out.append({
            "start": seg.index[0].isoformat(),
            "end": seg.index[-1].isoformat(),
            "std": seg_std,
            "mean": seg_mean,
            "cv": seg_cv,
        })

    return out


# =========================
# Per-machine Analyzer v2
# =========================

def analyze_trend_per_machine_v2(
    df: pd.DataFrame,
    intent: dict,
    ts_col: str = "ts",
    machine_col: str = "machine_id",
    value_col: str = "value",
    agg: str = "mean",
    z_th: float = 3.0,      # ambang global z-score
    rz_th: float = 3.0,     # ambang rolling z-score
    k_mad: float = 4.5,     # ambang MAD robust
    cv_th: float = 0.10,    # flag fluktuasi besar jika CV > ini
    roll_win_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Analisis per-mesin:
    - min/max raw dan resampled
    - tren (arah, %)
    - volatility (std, cv)
    - outliers (global z, rolling z, MAD)
    - periode fluktuasi tertinggi
    - ringkasan gabungan
    """

    if roll_win_map is None:
        roll_win_map = {"1H": 24, "1D": 7, "1W-MON": 6, "1MS": 3}

    start = intent["time_range"]["start"]
    end = intent["time_range"]["end"]
    scope = intent["scope"]
    gran = (intent.get("granularity") or "DAY").upper()
    machines = (intent.get("machines") or [])

    # filter data sesuai intent
    df1 = df.copy()
    df1[ts_col] = pd.to_datetime(df1[ts_col], errors="coerce")
    df1 = df1.dropna(subset=[ts_col, value_col])

    if start:
        df1 = df1[df1[ts_col] >= pd.Timestamp(start)]
    if end:
        df1 = df1[df1[ts_col] <= pd.Timestamp(end)]
    if scope == "SUBSET" and machines:
        df1 = df1[df1[machine_col].isin(machines)]

    if df1.empty:
        return {
            "empty": True,
            "reason": "No data in range/selection",
            "intent": intent,
            "machines": {},
            "combined": {
                "any_fluctuation": False,
                "machines_with_outlier": [],
                "max_cv": 0.0,
            },
            "freq": None,
        }

    # pilih freq resample
    pref = {
        "HOUR": "1H",
        "DAY": "1D",
        "WEEK": "1W-MON",
        "MONTH": "1D",   # daily view for monthly queries
        "YEAR": "1MS",
        "ALL": "1D",
    }.get(gran, "1D")

    auto = _auto_freq(start, end)
    order = ["1H", "1D", "1W-MON", "1MS", "1Y"]
    # pilih frekuensi yang lebih halus (lebih "kecil" di order)
    freq = pref if order.index(pref) <= order.index(auto) else auto
    rwin = roll_win_map.get(freq, 7)

    results: Dict[str, Any] = {}

    for mid, g0 in df1.groupby(machine_col):
        g0 = g0.sort_values(ts_col)
        s_raw = g0.set_index(ts_col)[value_col].astype(float)

        # min/max RAW
        vmin_raw = float(s_raw.min())
        vmax_raw = float(s_raw.max())
        tmin_raw = s_raw.idxmin().isoformat()
        tmax_raw = s_raw.idxmax().isoformat()

        # resample untuk analisis tren makro
        s = s_raw.resample(freq).agg(agg).dropna()
        if s.empty:
            results[mid] = {"empty": True}
            continue

        # tren
        tstats = _trend_stats(s)

        # min/max RESAMPLED
        vmin_res = float(s.min())
        vmax_res = float(s.max())
        tmin_res = s.idxmin().isoformat()
        tmax_res = s.idxmax().isoformat()

        # volatilitas makro
        std_val = float(s.std(ddof=0))
        mean_val = float(s.mean())
        cv_val = float(std_val / mean_val) if mean_val != 0 else 0.0

        # --- OUTLIERS di RAW ---
        # global z-score
        mu = s_raw.mean()
        sig = s_raw.std(ddof=0)
        sig = sig if sig and sig != 0 else 1.0
        z = (s_raw - mu) / sig

        # robust MAD
        med, madv = _mad(s_raw)
        r_robust = 0.6745 * (s_raw - med) / (madv or 1.0)

        # rolling z (spike lokal)
        rz = _rolling_z(s_raw, win=rwin)

        mask = (z.abs() >= z_th) | (r_robust.abs() >= k_mad) | (rz.abs() >= rz_th)

        outs: List[Dict[str, Any]] = []
        if mask.any():
            # skor gabungan supaya kita bisa ambil top N outlier yang paling "parah"
            score = np.maximum.reduce([
                z.abs().fillna(0).to_numpy(),
                r_robust.abs().fillna(0).to_numpy(),
                (rz.abs().fillna(0).to_numpy() if isinstance(rz, pd.Series) else np.zeros_like(z.values)),
            ])
            order_idx = np.argsort(-score)  # descending
            idx_list = mask.index.to_list()
            for i in order_idx[:50]:  # batas 50 titik ekstrem
                ts = idx_list[i]
                if not mask.loc[ts]:
                    continue
                outs.append({
                    "ts": ts.isoformat(),
                    "value": float(s_raw.loc[ts]),
                    "z": float(z.loc[ts]) if ts in z.index else None,
                    "rz": float(rz.loc[ts]) if isinstance(rz, pd.Series) and ts in rz.index else None,
                    "r": float(r_robust.loc[ts]) if ts in r_robust.index else None,
                })

        # periode fluktuasi tertinggi (pakai series resampled s)
        top_win = _top_fluctuation_windows(s, win=max(3, min(rwin, len(s))), k=3)

        # flag fluktuasi besar
        flg = (cv_val > cv_th) or (len(outs) >= 1)

        results[mid] = {
            "points": int(len(s)),
            "trend": tstats,
            "raw_min": {"value": vmin_raw, "ts": tmin_raw},
            "raw_max": {"value": vmax_raw, "ts": tmax_raw},
            "resampled_min": {"value": vmin_res, "ts": tmin_res},
            "resampled_max": {"value": vmax_res, "ts": tmax_res},
            "volatility": {"std": std_val, "mean": mean_val, "cv": cv_val},
            "outliers": {"count": len(outs), "points": outs},
            "fluctuation_windows": top_win,
            "fluctuation": flg,
            "freq_used": freq,
            "roll_window": rwin,
        }

    combined = {
        "any_fluctuation": any(
            v.get("fluctuation") for v in results.values() if not v.get("empty")
        ),
        "machines_with_outlier": [
            m
            for m, v in results.items()
            if (not v.get("empty")) and v["outliers"]["count"] > 0
        ],
        "max_cv": max(
            [v["volatility"]["cv"] for v in results.values() if not v.get("empty")],
            default=0.0,
        ),
    }

    return {
        "intent": intent,
        "freq": freq,
        "machines": results,
        "combined": combined,
    }


# =========================
# Pretty printer v2
# =========================

def pretty_per_machine_v2(
    report: Dict[str, Any],
    decimal_comma: bool = True,
    places: int = 6,
    show_outliers: int = 3,
    show_windows: int = 3,
) -> str:
    """
    Render teks ringkas per mesin + ringkasan gabungan.
    """
    if report.get("empty"):
        return f"Tidak ada data ({report.get('reason','')})"

    def fmt(x, p=6):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "NA"
        s = f"{x:.{p}f}"
        return s.replace(".", ",") if decimal_comma else s

    lines: List[str] = []

    for mid, r in report["machines"].items():
        if r.get("empty"):
            lines.append(f"[{mid}] kosong.")
            continue

        t = r["trend"]
        dir_map = {"up": "meningkat", "down": "menurun", "flat": "stabil", None: "-"}

        pct_disp = "NA"
        if t["pct"] is not None:
            pct_disp = fmt(t["pct"] * 100, places)  # jadi persen

        lines.append(
            f"[{mid}] tren {dir_map.get(t['direction'])} ({pct_disp}%). "
            f"CV={fmt(r['volatility']['cv'], places)}, "
            f"std={fmt(r['volatility']['std'], places)} "
            f"(freq={r['freq_used']}, roll_win={r['roll_window']})."
        )

        # min/max RAW + RESAMPLED
        lines.append(
            "  RAW  : "
            f"min={fmt(r['raw_min']['value'], places)} ({r['raw_min']['ts'][:19]}), "
            f"max={fmt(r['raw_max']['value'], places)} ({r['raw_max']['ts'][:19]})."
        )
        lines.append(
            "  RESMP: "
            f"min={fmt(r['resampled_min']['value'], places)} ({r['resampled_min']['ts'][:19]}), "
            f"max={fmt(r['resampled_max']['value'], places)} ({r['resampled_max']['ts'][:19]})."
        )

        # outliers detail
        pts = r["outliers"]["points"][:show_outliers]
        if pts:
            lines.append(f"  Outliers ({len(r['outliers']['points'])}):")
            for p in pts:
                lines.append(
                    "    - "
                    f"{p['ts'][:19]} | "
                    f"val={fmt(p['value'], places)} | "
                    f"z={fmt(p['z'],3)} | "
                    f"rz={fmt(p['rz'],3)} | "
                    f"r={fmt(p['r'],3)}"
                )
        else:
            lines.append("  Outliers: tidak terdeteksi.")

        # fluctuation windows
        wins = r["fluctuation_windows"][:show_windows]
        if wins:
            lines.append("  Periode fluktuasi tertinggi:")
            for w in wins:
                lines.append(
                    "    - "
                    f"{w['start'][:19]} → {w['end'][:19]} | "
                    f"std={fmt(w['std'],places)} | "
                    f"mean={fmt(w['mean'],places)} | "
                    f"cv={fmt(w['cv'],places)}"
                )

    comb = report["combined"]
    lines.append(
        "(GABUNGAN) "
        f"ada_fluktuasi={comb['any_fluctuation']}, "
        f"max_CV={fmt(comb['max_cv'], places)}, "
        f"mesin_dengan_outlier={comb['machines_with_outlier']}"
    )

    return "\n".join(lines)
# ======================================================================================
# CARA PAKAI
# main_demo.py

# from data_loader import build_df_raw
# from nlp_intent import interpret
# from machine_analyzer import analyze_trend_per_machine_v2, pretty_per_machine_v2

# # -------------------------------------------------
# # 1. Siapin data mentah dari Excel jadi df_raw
# # -------------------------------------------------
# df_raw, _ = build_df_raw(
#     excel_path="/workspace/Machine_Value_Processed.xlsx",  # path file Excel asli kamu
#     sheet_name="SW_3min",                                  # sheet yang berisi data mesin
#     save_csv="/workspace/df_raw.csv",                      # optional, boleh None
#     with_lag=False                                         # kita gak perlu fitur lag di analisis trend
# )

# # df_raw sekarang harus punya kolom:
# #   machine_id | ts | value
# # Contoh head:
# #   XP888A     | 2025-10-20 13:03:00 | 42.7
# #   XP888A     | 2025-10-20 13:06:00 | 43.1
# #   PQ667      | 2025-10-20 13:03:00 | 77.2
# # dst.

# # -------------------------------------------------
# # 2. Dapatkan intent dari natural language user
# # -------------------------------------------------
# # Misal operator nanya:
# user_query = "Cek tren mesin XP888A dan PQ667 bulan ini, ada fluktuasi besar gak?"

# # Kamu BUTUH output model NLP kamu (triple-head classifier)
# # karena di Colab tadi kamu pakai runner.predict(query)
# # Di sini aku bikin dummy biar contoh end-to-end jalan.

# model_pred = {
#     "scope": {"label": "SUBSET", "confidence": 0.92},
#     "time":  {"label": "DAY",    "confidence": 0.80},  # fallback granularity kalau parser gak ngerti
#     "case":  {"label": "SINGLE_PERIOD", "confidence": 0.75},
# }

# # interpret() akan:
# #  - parse waktu ("bulan ini")
# #  - parse mesin ("XP888A", "PQ667")
# #  - gabung sama prediksi model
# intent = interpret(user_query, model_pred)

# print("INTENT:")
# print(intent)
# # intent kurang lebih akan jadi bentuk:
# # {
# #   'query': 'Cek tren mesin XP888A ...',
# #   'scope': 'SUBSET',
# #   'machines': ['XP888A','PQ667'],
# #   'granularity': 'MONTH',
# #   'time_range': {
# #       'start': '2025-10-01T00:00:00', 
# #       'end':   '2025-10-23T23:59:59', 
# #       'note': 'bulan ini (MTD)'
# #   },
# #   'case': 'SINGLE_PERIOD',
# #   'route_hints': [...],
# #   'meta': {...}
# # }

# # -------------------------------------------------
# # 3. Jalankan analisis per mesin
# # -------------------------------------------------
# report = analyze_trend_per_machine_v2(
#     df_raw,
#     intent,
#     ts_col="ts",
#     machine_col="machine_id",
#     value_col="value",
# )

# # Struktur 'report' kurang lebih:
# # {
# #   'intent': {...},
# #   'freq': '1D',     # resample freq yg dipakai (misal harian)
# #   'machines': {
# #       'XP888A': {
# #           'points': ...,
# #           'trend': {'slope':..., 'pct':..., 'direction':'up/down/flat'},
# #           'raw_min': {...}, 'raw_max': {...},
# #           'resampled_min': {...}, 'resampled_max': {...},
# #           'volatility': {'std':..., 'mean':..., 'cv':...},
# #           'outliers': {'count':..., 'points':[ {...}, ... ]},
# #           'fluctuation_windows': [ {window paling liar}, ...],
# #           'fluctuation': True/False,
# #           'freq_used': '1D',
# #           'roll_window': 7
# #       },
# #       'PQ667': {...}
# #   },
# #   'combined': {
# #       'any_fluctuation': True/False,
# #       'machines_with_outlier': [...],
# #       'max_cv': ...
# #   }
# # }

# print("RAW REPORT (dict):")
# print(report)

# # -------------------------------------------------
# # 4. Buat teks laporan yang enak dibaca operator / engineer
# # -------------------------------------------------
# pretty_text = pretty_per_machine_v2(
#     report,
#     decimal_comma=True,   # kalau mau pakai koma sebagai desimal (Indonesia style)
#     places=4,             # jumlah digit desimal
#     show_outliers=3,      # tampilkan max 3 spike ekstrem per mesin
#     show_windows=2        # tampilkan 2 window volatilitas tertinggi
# )

# print("\nLAPORAN HUMAN-READABLE:\n")
# print(pretty_text)
# ======================================================================================


# import pandas as pd
# import numpy as np

# # =========================
# #  Utilities
# # =========================
# def _auto_freq(start, end):
#     if start is None or end is None:
#         return "1D"
#     span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
#     if span <= 3*24*3600:      return "1H"     # ≤ 3 hari
#     if span <= 35*24*3600:     return "1D"     # ≤ 5 minggu
#     if span <= 200*24*3600:    return "1W-MON" # ≤ ~6.5 bulan
#     return "1MS"                               

# def _trend_stats(series: pd.Series):
#     if len(series) < 2:
#         return {"slope": 0.0, "pct": None, "direction": None}
#     y = series.values.astype(float)
#     x = np.arange(len(series), dtype=float)
#     varx = np.var(x)
#     slope = float(np.cov(x, y, bias=True)[0,1] / (varx if varx != 0 else 1))
#     split = max(1, int(len(series)*0.7))
#     prev = series.iloc[:split]
#     recent = series.iloc[split:]
#     pct = ((recent.mean()-prev.mean())/abs(prev.mean())) if prev.mean() not in (0, np.nan, None, 0.0) else None
#     direction = "flat"
#     if pct is not None:
#         if pct > 0.02 or slope > 1e-4: direction = "up"
#         if pct < -0.02 or slope < -1e-4: direction = "down"
#         if abs(pct) <= 0.01 and abs(slope) <= 1e-4: direction = "flat"
#     return {"slope": float(slope), "pct": None if pct is None else float(pct), "direction": direction}

# def _mad(x: pd.Series):
#     med = x.median()
#     mad = (x - med).abs().median()
#     return med, (mad if mad and mad != 0 else 1.0)

# def _rolling_z(series: pd.Series, win: int = 12):
#     """rolling z-score (mean/std di-window), tahan seasonal kecil"""
#     if len(series) < max(5, win):
#         return pd.Series(index=series.index, dtype=float)
#     m = series.rolling(win, min_periods=max(3, win//3)).mean()
#     s = series.rolling(win, min_periods=max(3, win//3)).std(ddof=0).replace(0, np.nan)
#     return (series - m) / s

# def _top_fluctuation_windows(series: pd.Series, win: int, k: int = 3):
#     """cari periode dengan volatilitas (std/CV) tertinggi"""
#     if len(series) < max(5, win):
#         return []
#     roll_std = series.rolling(win, min_periods=max(3, win//3)).std(ddof=0)
#     roll_mean = series.rolling(win, min_periods=max(3, win//3)).mean().replace(0, np.nan)
#     roll_cv = (roll_std / roll_mean).replace([np.inf, -np.inf], np.nan)
#     # ambil top-k window center
#     sc = roll_cv.fillna(0)
#     idxs = sc.nlargest(k).index
#     out = []
#     half = max(1, win//2)
#     for ts in idxs:
#         try:
#             s = series.loc[:ts].iloc[-win:]
#         except Exception:
#             continue
#         if s.empty: 
#             continue
#         out.append({
#             "start": s.index[0].isoformat(),
#             "end": s.index[-1].isoformat(),
#             "std": float(s.std(ddof=0)),
#             "mean": float(s.mean()),
#             "cv": float((s.std(ddof=0) / (s.mean() if s.mean()!=0 else np.nan)) if s.mean()!=0 else np.nan),
#         })
#     return out

# # =========================
# #  Per-machine Analyzer v2
# # =========================
# def analyze_trend_per_machine_v2(
#     df: pd.DataFrame,
#     intent: dict,
#     ts_col="ts",
#     machine_col="machine_id",
#     value_col="value",
#     agg="mean",
#     z_th=3.0,          # global z-score ambang
#     rz_th=3.0,         # rolling z-score ambang
#     k_mad=4.5,         # robust MAD ambang
#     cv_th=0.10,        # fluktuasi besar jika CV > ini
#     roll_win_map=None  # map frekuensi -> window
# ):
#     """
#     Analisis per-mesin yang komprehensif:
#     - min/max RAW (langsung dari data) + min/max RESAMPLED
#     - outlier multi-metode: z-score global, rolling z-score, robust MAD
#     - periode fluktuasi (top-k jendela CV tertinggi)
#     - tren (arah & %)
#     """
#     if roll_win_map is None:
#         roll_win_map = {"1H": 24, "1D": 7, "1W-MON": 6, "1MS": 3}

#     start = intent["time_range"]["start"]
#     end   = intent["time_range"]["end"]
#     scope = intent["scope"]
#     gran  = (intent.get("granularity") or "DAY").upper()
#     machines = (intent.get("machines") or [])

#     df1 = df.copy()
#     df1[ts_col] = pd.to_datetime(df1[ts_col], errors="coerce")
#     df1 = df1.dropna(subset=[ts_col, value_col])
#     if start: df1 = df1[df1[ts_col] >= pd.Timestamp(start)]
#     if end:   df1 = df1[df1[ts_col] <= pd.Timestamp(end)]
#     if scope == "SUBSET" and machines:
#         df1 = df1[df1[machine_col].isin(machines)]
#     if df1.empty:
#         return {"empty": True, "reason": "No data in range/selection", "intent": intent}

#     pref = {"HOUR":"1H","DAY":"1D","WEEK":"1W-MON","MONTH":"1D","YEAR":"1MS","ALL":"1D"}.get(gran, "1D")
#     auto = _auto_freq(start, end)
#     order = ["1H","1D","1W-MON","1MS","1Y"]
#     freq = pref if order.index(pref) <= order.index(auto) else auto
#     rwin = roll_win_map.get(freq, 7)

#     results = {}
#     for mid, g0 in df1.groupby(machine_col):
#         g0 = g0.sort_values(ts_col)
#         s_raw = g0.set_index(ts_col)[value_col].astype(float)

#         # ---------- min/max RAW (akurasi penuh)
#         vmin_raw, vmax_raw = float(s_raw.min()), float(s_raw.max())
#         tmin_raw, tmax_raw = s_raw.idxmin().isoformat(), s_raw.idxmax().isoformat()

#         # ---------- resample untuk tren & fluktuasi makro
#         s = s_raw.resample(freq).agg(agg).dropna()
#         if s.empty:
#             results[mid] = {"empty": True}
#             continue

#         # tren
#         tstats = _trend_stats(s)

#         # min/max RESAMPLED (berguna untuk laporan periodik)
#         vmin_res, vmax_res = float(s.min()), float(s.max())
#         tmin_res, tmax_res = s.idxmin().isoformat(), s.idxmax().isoformat()

#         # volatilitas makro
#         std = float(s.std(ddof=0))
#         mean = float(s.mean())
#         cv = float(std/mean) if mean != 0 else 0.0

#         # ---------- OUTLIER multi-metode di RAW (lebih sensitif)
#         # z-score global
#         mu, sig = s_raw.mean(), s_raw.std(ddof=0)
#         sig = sig if sig and sig != 0 else 1.0
#         z = (s_raw - mu) / sig

#         # robust MAD
#         med, madv = _mad(s_raw)
#         r = 0.6745 * (s_raw - med) / (madv or 1.0)

#         # rolling z-score (deteksi spike lokal)
#         rz = _rolling_z(s_raw, win=rwin)

#         mask = (z.abs() >= z_th) | (r.abs() >= k_mad) | (rz.abs() >= rz_th)
#         outs = []
#         if mask.any():
#             # urutkan berdasarkan skor terbesar (gabungkan magnitude terbaik dari ketiganya)
#             score = np.maximum.reduce([
#                 z.abs().fillna(0).values,
#                 r.abs().fillna(0).values,
#                 (rz.abs().fillna(0).values if isinstance(rz, pd.Series) else np.zeros_like(z.values))
#             ])
#             order_idx = np.argsort(-score)  # descending
#             idx = mask.index.to_list()
#             for i in order_idx[:50]:  # batas 50 titik
#                 ts = idx[i]
#                 if not mask.loc[ts]:
#                     continue
#                 outs.append({
#                     "ts": ts.isoformat(),
#                     "value": float(s_raw.loc[ts]),
#                     "z": float(z.loc[ts]) if ts in z.index else None,
#                     "rz": float(rz.loc[ts]) if (isinstance(rz, pd.Series) and ts in rz.index) else None,
#                     "r": float(r.loc[ts]) if ts in r.index else None
#                 })

#         # ---------- Periode fluktuasi (top windows)
#         top_win = _top_fluctuation_windows(s, win=max(3, min(rwin, len(s))), k=3)

#         # flag fluktuasi besar per-mesin
#         flg = (cv > cv_th) or (len(outs) >= 1)

#         results[mid] = {
#             "points": int(len(s)),
#             "trend": tstats,
#             "raw_min": {"value": vmin_raw, "ts": tmin_raw},
#             "raw_max": {"value": vmax_raw, "ts": tmax_raw},
#             "resampled_min": {"value": vmin_res, "ts": tmin_res},
#             "resampled_max": {"value": vmax_res, "ts": tmax_res},
#             "volatility": {"std": std, "mean": mean, "cv": cv},
#             "outliers": {"count": len(outs), "points": outs},
#             "fluctuation_windows": top_win,
#             "fluctuation": flg,
#             "freq_used": freq,
#             "roll_window": rwin
#         }

#     combined = {
#         "any_fluctuation": any(v.get("fluctuation") for v in results.values() if not v.get("empty")),
#         "machines_with_outlier": [m for m,v in results.items() if (not v.get("empty")) and v["outliers"]["count"]>0],
#         "max_cv": max([v["volatility"]["cv"] for v in results.values() if not v.get("empty")], default=0.0)
#     }
#     return {"intent": intent, "freq": freq, "machines": results, "combined": combined}

# # =========================
# #  Pretty printer v2
# # =========================
# def pretty_per_machine_v2(report: dict, decimal_comma=True, places=6, show_outliers=3, show_windows=3):
#     if report.get("empty"):
#         return f"Tidak ada data ({report.get('reason','')})"
#     def fmt(x, p=6):
#         if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))): return "NA"
#         s = f"{x:.{p}f}"
#         return s.replace(".", ",") if decimal_comma else s
#     lines = []
#     for mid, r in report["machines"].items():
#         if r.get("empty"):
#             lines.append(f"[{mid}] kosong.")
#             continue
#         t = r["trend"]; dir_map = {"up":"meningkat","down":"menurun","flat":"stabil", None:"-"}
#         lines.append(
#             f"[{mid}] tren {dir_map.get(t['direction'])} ({fmt((t['pct'] or 0)*100, places)}%). "
#             f"CV={fmt(r['volatility']['cv'], places)}, std={fmt(r['volatility']['std'], places)} "
#             f"(freq={r['freq_used']}, roll_win={r['roll_window']})."
#         )
#         # min/max RAW (lebih akurat) + resampled
#         lines.append(
#             f"  RAW  : min={fmt(r['raw_min']['value'], places)} ({r['raw_min']['ts'][:19]}), "
#             f"max={fmt(r['raw_max']['value'], places)} ({r['raw_max']['ts'][:19]})."
#         )
#         lines.append(
#             f"  RESMP: min={fmt(r['resampled_min']['value'], places)} ({r['resampled_min']['ts'][:19]}), "
#             f"max={fmt(r['resampled_max']['value'], places)} ({r['resampled_max']['ts'][:19]})."
#         )
#         # outliers detail
#         pts = r['outliers']['points'][:show_outliers]
#         if pts:
#             lines.append(f"  Outliers ({len(r['outliers']['points'])}):")
#             for p in pts:
#                 lines.append(
#                     f"    - {p['ts'][:19]} | val={fmt(p['value'], places)} | z={fmt(p['z'],3)} | rz={fmt(p['rz'],3)} | r={fmt(p['r'],3)}"
#                 )
#         else:
#             lines.append("  Outliers: tidak terdeteksi.")
#         # fluctuation windows
#         wins = r["fluctuation_windows"][:show_windows]
#         if wins:
#             lines.append("  Periode fluktuasi tertinggi:")
#             for w in wins:
#                 lines.append(
#                     f"    - {w['start'][:19]} → {w['end'][:19]} | std={fmt(w['std'],places)} | mean={fmt(w['mean'],places)} | cv={fmt(w['cv'],places)}"
#                 )
#     comb = report["combined"]
#     lines.append(
#         f"(GABUNGAN) ada_fluktuasi={comb['any_fluctuation']}, max_CV={fmt(comb['max_cv'], places)}, "
#         f"mesin_dengan_outlier={comb['machines_with_outlier']}"
#     )
#     return "\n".join(lines)
