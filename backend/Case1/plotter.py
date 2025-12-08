import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

def _ensure_dir(p: str) -> None:
    """
    Pastikan folder untuk file path `p` ada.
    """
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(x):
    """
    Ubah "12,5" -> 12.5 dan safe-coerce selainnya ke float.
    """
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


# =========================
# Plotters
# =========================

def plot_raw_series(
    df_raw: pd.DataFrame,
    machine_id: str,
    *,
    ts_col: str = "ts",
    val_col: str = "value",
    machine_col: str = "machine_id",
    save_path: Optional[str] = None,
    title_suffix: str = "RAW",
) -> Optional[str]:
    """
    Plot time series mentah (tanpa resample) untuk satu mesin.
    Return path gambar (kalau save_path dikasih) atau None kalau gak jadi nge-plot.
    """
    d = df_raw[df_raw[machine_col] == machine_id].copy()
    if d.empty:
        return None

    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d[val_col] = d[val_col].apply(_coerce_numeric)
    d = d.dropna(subset=[ts_col, val_col]).sort_values(ts_col)
    if d.empty:
        return None

    plt.figure()
    plt.plot(d[ts_col], d[val_col], marker=".", linestyle="-")
    plt.xlabel("Waktu")
    plt.ylabel(val_col)
    plt.title(f"{machine_id} — {title_suffix}")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=140)
        plt.close()
        return save_path

    plt.close()
    return None


def plot_resampled_series(
    df_raw: pd.DataFrame,
    machine_id: str,
    *,
    freq: str = "1D",
    agg: str = "mean",
    ts_col: str = "ts",
    val_col: str = "value",
    machine_col: str = "machine_id",
    save_path: Optional[str] = None,
    title_suffix: str = "RESAMPLED",
) -> Optional[str]:
    """
    Plot time series mesin setelah di-resample (daily/hourly/etc).
    """
    d = df_raw[df_raw[machine_col] == machine_id].copy()
    if d.empty:
        return None

    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d[val_col] = d[val_col].apply(_coerce_numeric)
    d = d.dropna(subset=[ts_col, val_col]).sort_values(ts_col)
    if d.empty:
        return None

    # Jadikan DatetimeIndex supaya resample jalan
    s = d.set_index(ts_col)[val_col].astype(float)
    s.index = pd.DatetimeIndex(s.index)

    s = s.resample(freq).agg(agg).dropna()
    if s.empty:
        return None

    plt.figure()
    plt.plot(s.index, s.values, marker="o", linestyle="-")
    plt.xlabel("Waktu")
    plt.ylabel(val_col)
    plt.title(f"{machine_id} — {title_suffix} ({freq})")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=140)
        plt.close()
        return save_path

    plt.close()
    return None


def _plot_histogram_if_possible(
    df_raw: pd.DataFrame,
    machine_id: str,
    *,
    ts_col: str,
    val_col: str,
    machine_col: str,
    save_path: str,
) -> Optional[str]:
    """
    Bikin histogram distribusi value per mesin (kalau datanya cukup).
    Return path file PNG atau None kalau gak dibuat.
    """
    vals = df_raw.loc[df_raw[machine_col] == machine_id, val_col]
    if vals.empty:
        return None

    vals = vals.apply(_coerce_numeric).dropna()
    if len(vals) < 2:
        return None

    plt.figure()
    plt.hist(vals, bins=30)
    plt.xlabel(val_col)
    plt.ylabel("Count")
    plt.title(f"{machine_id} — Histogram")
    plt.tight_layout()

    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=140)
    plt.close()

    return save_path


# =========================
# Metadata packer for LLM / downstream
# =========================

def package_metadata_for_llm(
    report: Dict[str, Any],
    limit_outliers: int = 5,
) -> Dict[str, Any]:
    """
    Ringkas output analyze_trend_per_machine_v2() jadi dict yang:
    - gampang dikonsumsi LLM untuk bikin narasi
    - gampang disimpan jadi JSON

    report bentuknya:
        {
            "intent": {...},
            "freq": "1D",
            "machines": {
                "XP888A": {
                    "trend": {...},
                    "volatility": {...},
                    "outliers": {"count": X, "points": [...]},
                    ...
                },
                ...
            },
            "combined": {...}
        }
    """
    meta: Dict[str, Any] = {
        "freq": report.get("freq"),
        "any_fluctuation": report.get("combined", {}).get("any_fluctuation"),
        "max_cv": report.get("combined", {}).get("max_cv"),
        "machines": [],
    }

    for mid, mrep in report.get("machines", {}).items():
        if mrep.get("empty"):
            meta["machines"].append({
                "machine_id": mid,
                "empty": True,
            })
            continue

        # potong daftar outliers biar ga kebanyakan
        out_points = (mrep.get("outliers", {}).get("points") or [])[:limit_outliers]

        meta["machines"].append({
            "machine_id": mid,
            "points": mrep.get("points"),
            "trend_direction": mrep.get("trend", {}).get("direction"),
            "trend_pct_change": mrep.get("trend", {}).get("pct"),
            "cv": mrep.get("volatility", {}).get("cv"),
            "std": mrep.get("volatility", {}).get("std"),
            "raw_min": mrep.get("raw_min"),
            "raw_max": mrep.get("raw_max"),
            "resampled_min": mrep.get("resampled_min"),
            "resampled_max": mrep.get("resampled_max"),
            "outliers_count": mrep.get("outliers", {}).get("count"),
            "outliers_examples": out_points,
            "fluctuation_windows": mrep.get("fluctuation_windows"),
            "freq_used": mrep.get("freq_used"),
            "roll_window": mrep.get("roll_window"),
            "fluctuation_flag": mrep.get("fluctuation"),
        })

    return meta


# =========================
# Dispatcher
# =========================

def serve_insight_with_charts(
    df_raw: pd.DataFrame,
    intent: Dict[str, Any],
    report_fn,
    subintent: str = "summary",
    out_dir: str = "./plots",
    ts_col: str = "ts",
    machine_col: str = "machine_id",
    value_col: str = "value",
) -> Dict[str, Any]:
    """
    High-level helper untuk:
    1. Run analyzer (misal analyze_trend_per_machine_v2)
    2. Build metadata ringkas buat LLM / dashboard
    3. Generate plot per mesin:
        - raw timeseries
        - resampled timeseries
        - histogram distribusi
    4. Return path file PNG supaya gampang dipakai di UI

    Return dict seperti:
    {
        "meta": {...},
        "plot_paths": {
            "XP888A": ["./plots/XP888A_raw.png", "./plots/XP888A_resampled_1D.png", "./plots/XP888A_hist.png"],
            "PQ667": [...],
        },
        "skipped_plots": False
    }
    """
    si = (subintent or "summary").lower()
    only_minmax = si in {"min_only", "max_only", "minmax"}

    report = report_fn(
        df_raw,
        intent,
        ts_col=ts_col,
        machine_col=machine_col,
        value_col=value_col,
    )

    if report.get("empty"):
        # analyzer bilang gak ada data
        return {
            "empty": True,
            "reason": report.get("reason", "no data"),
        }

    meta = package_metadata_for_llm(report, limit_outliers=5)

    # Kalau user cuma minta min/max, gak perlu buang waktu bikin plot
    if only_minmax:
        return {
            "meta": meta,
            "plot_paths": {},
            "skipped_plots": True,
        }

    plot_paths: Dict[str, List[str]] = {}
    freq_used = report.get("freq", "1D")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for mid in report["machines"].keys():
        # raw line plot
        raw_path = os.path.join(out_dir, f"{mid}_raw.png")
        p1 = plot_raw_series(
            df_raw,
            mid,
            ts_col=ts_col,
            val_col=value_col,
            machine_col=machine_col,
            save_path=raw_path,
            title_suffix="RAW",
        )

        # resampled plot
        res_path = os.path.join(out_dir, f"{mid}_resampled_{freq_used}.png")
        p2 = plot_resampled_series(
            df_raw,
            mid,
            freq=freq_used,
            agg="mean",
            ts_col=ts_col,
            val_col=value_col,
            machine_col=machine_col,
            save_path=res_path,
            title_suffix="RESAMPLED",
        )

        # histogram
        hist_path = os.path.join(out_dir, f"{mid}_hist.png")
        p3 = _plot_histogram_if_possible(
            df_raw,
            mid,
            ts_col=ts_col,
            val_col=value_col,
            machine_col=machine_col,
            save_path=hist_path,
        )

        # kumpulkan hanya path yang berhasil dibuat (not None)
        plot_paths[mid] = [p for p in [p1, p2, p3] if p]

    return {
        "meta": meta,
        "plot_paths": plot_paths,
        "skipped_plots": False,
    }

# =======================================================================================
# CARA PAKAI
# from data_loader import build_df_raw
# from nlp_intent import interpret
# from machine_analyzer import analyze_trend_per_machine_v2, pretty_per_machine_v2
# from plotter import serve_insight_with_charts

# # 1. load data
# df_raw, _ = build_df_raw(
#     excel_path="/workspace/Machine_Value_Processed.xlsx",
#     sheet_name="SW_3min",
#     with_lag=False,
# )

# # 2. bikin intent dari query user + prediksi model
# user_query = "Tolong cek tren semua mesin minggu ini, highlight outlier"
# model_pred = {
#     "scope": {"label": "ALL", "confidence": 0.9},
#     "time":  {"label": "DAY", "confidence": 0.8},
#     "case":  {"label": "SINGLE_PERIOD", "confidence": 0.7},
# }
# intent = interpret(user_query, model_pred)

# # 3. analisis per mesin
# report = analyze_trend_per_machine_v2(df_raw, intent)

# # 4. teks ringkas buat operator
# print(pretty_per_machine_v2(report))

# # 5. plus grafik dan metadata LLM
# bundle = serve_insight_with_charts(
#     df_raw,
#     intent,
#     report_fn=analyze_trend_per_machine_v2,
#     subintent="summary",
#     out_dir="./plots"
# )

# print("META UNTUK LLM / DASHBOARD:")
# print(bundle["meta"])
# print("PATH GAMBAR:")
# print(bundle["plot_paths"])
# =======================================================================================
# CARA PAKAI
# from data_loader import build_df_raw
# from nlp_intent import interpret
# from machine_analyzer import analyze_trend_per_machine_v2, pretty_per_machine_v2
# from plotter import serve_insight_with_charts

# # 1. load data
# df_raw, _ = build_df_raw(
#     excel_path="/workspace/Machine_Value_Processed.xlsx",
#     sheet_name="SW_3min",
#     with_lag=False,
# )

# # 2. bikin intent dari query user + prediksi model
# user_query = "Tolong cek tren semua mesin minggu ini, highlight outlier"
# model_pred = {
#     "scope": {"label": "ALL", "confidence": 0.9},
#     "time":  {"label": "DAY", "confidence": 0.8},
#     "case":  {"label": "SINGLE_PERIOD", "confidence": 0.7},
# }
# intent = interpret(user_query, model_pred)

# # 3. analisis per mesin
# report = analyze_trend_per_machine_v2(df_raw, intent)

# # 4. teks ringkas buat operator
# print(pretty_per_machine_v2(report))

# # 5. plus grafik dan metadata LLM
# bundle = serve_insight_with_charts(
#     df_raw,
#     intent,
#     report_fn=analyze_trend_per_machine_v2,
#     subintent="summary",
#     out_dir="./plots"
# )

# print("META UNTUK LLM / DASHBOARD:")
# print(bundle["meta"])
# print("PATH GAMBAR:")

# =======================================================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from pathlib import Path

# def _ensure_dir(p):
#     Path(p).parent.mkdir(parents=True, exist_ok=True)

# def _coerce_numeric(x):
#     # dukung koma desimal
#     if isinstance(x, str):
#         x = x.replace(",", ".")
#     return pd.to_numeric(x, errors="coerce")

# def plot_raw_series(df_raw, machine_id, *,
#                     ts_col="ts", val_col="value", machine_col="machine_id",
#                     save_path=None, title_suffix="RAW"):
#     d = df_raw[df_raw[machine_col] == machine_id].copy()
#     if d.empty: return None
#     d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
#     d[val_col] = d[val_col].apply(_coerce_numeric)
#     d = d.dropna(subset=[ts_col, val_col]).sort_values(ts_col)
#     if d.empty: return None

#     plt.figure()
#     plt.plot(d[ts_col], d[val_col], marker=".", linestyle="-")
#     plt.xlabel("Waktu"); plt.ylabel(val_col); plt.title(f"{machine_id} — {title_suffix}")
#     plt.tight_layout()
#     if save_path:
#         _ensure_dir(save_path); plt.savefig(save_path, dpi=140); plt.close(); return save_path
#     return None

# def plot_resampled_series(df_raw, machine_id, *,
#                           freq="1D", agg="mean",
#                           ts_col="ts", val_col="value", machine_col="machine_id",
#                           save_path=None, title_suffix="RESAMPLED"):
#     d = df_raw[df_raw[machine_col] == machine_id].copy()
#     if d.empty: return None

#     # pastikan datetime index + numeric value
#     d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
#     d[val_col] = d[val_col].apply(_coerce_numeric)
#     d = d.dropna(subset=[ts_col, val_col]).sort_values(ts_col)
#     if d.empty: return None

#     s = d.set_index(ts_col)[val_col].astype(float)
#     # <- inilah kunci: paksa menjadi DatetimeIndex
#     s.index = pd.DatetimeIndex(s.index)
#     s = s.resample(freq).agg(agg).dropna()
#     if s.empty: return None

#     plt.figure()
#     plt.plot(s.index, s.values, marker="o", linestyle="-")
#     plt.xlabel("Waktu"); plt.ylabel(val_col); plt.title(f"{machine_id} — {title_suffix} ({freq})")
#     plt.tight_layout()
#     if save_path:
#         _ensure_dir(save_path); plt.savefig(save_path, dpi=140); plt.close(); return save_path
#     return None

# # --- di dispatcher, pastikan param 'machine_col' diteruskan ke plotters ---
# def serve_insight_with_charts(
#     df_raw: pd.DataFrame,
#     intent: dict,
#     report_fn,            # analyze_trend_per_machine_v2
#     subintent: str = "summary",
#     out_dir: str = "./plots",
#     ts_col="ts", machine_col="machine_id", value_col="value"
# ):
#     si = (subintent or "summary").lower()
#     only_minmax = si in {"min_only","max_only","minmax"}

#     report = report_fn(df_raw, intent, ts_col=ts_col, machine_col=machine_col, value_col=value_col)
#     if report.get("empty"):
#         return {"empty": True, "reason": report.get("reason", "no data")}

#     meta = package_metadata_for_llm(report, limit_outliers=5)

#     if only_minmax:
#         return {"meta": meta, "plot_paths": {}, "skipped_plots": True}

#     plot_paths = {}
#     freq_used = report.get("freq", "1D")
#     Path(out_dir).mkdir(parents=True, exist_ok=True)

#     for mid in report["machines"].keys():
#         p_raw = os.path.join(out_dir, f"{mid}_raw.png")
#         p1 = plot_raw_series(df_raw, mid, ts_col=ts_col, val_col=value_col, machine_col=machine_col,
#                              save_path=p_raw, title_suffix="RAW")

#         p_res = os.path.join(out_dir, f"{mid}_resampled_{freq_used}.png")
#         p2 = plot_resampled_series(df_raw, mid, freq=freq_used, agg="mean",
#                                    ts_col=ts_col, val_col=value_col, machine_col=machine_col,
#                                    save_path=p_res, title_suffix="RESAMPLED")

#         p_hist = os.path.join(out_dir, f"{mid}_hist.png")
#         # histogram opsional; aman skip bila <2 nilai
#         vals = df_raw.loc[df_raw[machine_col]==mid, value_col]
#         if not vals.empty:
#             vals = vals.apply(_coerce_numeric).dropna()
#             if len(vals) >= 2:
#                 plt.figure()
#                 plt.hist(vals, bins=30)
#                 plt.xlabel(value_col); plt.ylabel("Count"); plt.title(f"{mid} — Histogram")
#                 plt.tight_layout()
#                 _ensure_dir(p_hist); plt.savefig(p_hist, dpi=140); plt.close()
#                 p3 = p_hist
#             else:
#                 p3 = None
#         else:
#             p3 = None

#         plot_paths[mid] = [p for p in [p1, p2, p3] if p]

#     return {"meta": meta, "plot_paths": plot_paths, "skipped_plots": False}
