# ============================================================
# anomaly_range_pipeline.py
# Pipeline Anomali: nilai keluar dari rentang min/max per mesin
# ============================================================

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Utils internal
# ------------------------------------------------------------
def _coerce_num(x) -> float | int | None:
    """
    Konversi string '123,45' -> 123.45.
    Kalau gagal parse jadi NaN.
    """
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def _ensure_dir(p: str | Path) -> None:
    """
    Pastikan folder untuk path 'p' sudah ada.
    p bisa file path -> kita buat parent-nya.
    """
    Path(p).parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1) Load threshold tabel (min/max per machine)
# ------------------------------------------------------------
def load_thresholds(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Baca file batas min/max per mesin.
    path: .xlsx/.xls/.csv
    sheet: opsional (kalau Excel multi-sheet)

    Wajib ada kolom yg bisa dipetakan ke:
        - machine_id  (alias: tag number / machine / mesin / tag)
        - max_value   (alias: maximum / max / upper)
        - min_value   (alias: minimum / min / lower)

    Returns
    -------
    DataFrame[ machine_id:str, max_value:float, min_value:float ]
    """
    # load file
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # normalisasi header -> cari kolom mana yang mana
    def norm(s: str) -> str:
        return (
            (str(s) or "")
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

    cols_norm = {c: norm(c) for c in df.columns}

    col_tag = None
    for c, n in cols_norm.items():
        if n in {"tagnumber", "tagnumber.", "tag", "machine", "mesin"}:
            col_tag = c
            break

    col_max = None
    for c, n in cols_norm.items():
        if "max" in n:  # "max", "maximum", "maxvalue", etc.
            col_max = c
            break

    col_min = None
    for c, n in cols_norm.items():
        if "min" in n:  # "min", "minimum", "minvalue", etc.
            col_min = c
            break

    if not (col_tag and col_min and col_max):
        raise ValueError(
            f"Kolom wajib tidak ditemukan. Kolom ada: {list(df.columns)}. "
            "Harus ada semantik tag/machine + min + max."
        )

    out = df[[col_tag, col_max, col_min]].copy()
    out.columns = ["machine_id", "max_value", "min_value"]

    # bersihin tipe
    out["machine_id"] = out["machine_id"].astype(str).str.strip().str.upper()
    out["max_value"] = out["max_value"].apply(_coerce_num)
    out["min_value"] = out["min_value"].apply(_coerce_num)

    # drop yg kosong
    out = out.dropna(subset=["machine_id", "max_value", "min_value"]).reset_index(drop=True)

    # kalau min > max (data salah ketuk), swap
    swap_mask = out["min_value"] > out["max_value"]
    if swap_mask.any():
        tmp = out.loc[swap_mask, "min_value"].copy()
        out.loc[swap_mask, "min_value"] = out.loc[swap_mask, "max_value"]
        out.loc[swap_mask, "max_value"] = tmp

    return out


def build_threshold_map(df_thresh: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Ubah DataFrame threshold jadi dict:
        { "XP888A": (min_value, max_value), ... }
    """
    return {
        str(row["machine_id"]): (float(row["min_value"]), float(row["max_value"]))
        for _, row in df_thresh.iterrows()
    }


# ------------------------------------------------------------
# 2) Core detection: tandai baris yg keluar [min,max]
# ------------------------------------------------------------
def detect_range_anomalies(
    df_raw: pd.DataFrame,
    thresholds: Dict[str, Tuple[float, float]],
    *,
    ts_col: str = "ts",
    machine_col: str = "machine_id",
    value_col: str = "value",
    start: Optional[str] = None,
    end: Optional[str] = None,
    machines: Optional[List[str]] = None,
    fallback: Literal["infer", "skip"] = "infer",
) -> pd.DataFrame:
    """
    - Filter df_raw by waktu (start/end) & mesin (machines)
    - Untuk tiap row, cari batas expected_min/expected_max
      dari thresholds[ machine_id ].
      Kalau mesin tidak terdaftar di thresholds:
        * "skip"  -> expected_min/max = NaN
        * "infer" -> hitung quantile(1%,99%) atau min/max lokal mesin tsb.

    Return DataFrame copy dengan tambahan kolom:
      expected_min, expected_max,
      is_anomaly (bool),
      which_bound ('below'|'above'|None),
      distance        (seberapa jauh dari batas terdekat),
      distance_pct    (relative distance terhadap lebar range).
    """
    df = df_raw.copy()

    # coerces
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df[value_col] = df[value_col].apply(_coerce_num)
    df[machine_col] = df[machine_col].astype(str).str.upper().str.strip()

    # buang baris ga valid
    df = df.dropna(subset=[ts_col, machine_col, value_col])

    # filter waktu
    if start:
        df = df[df[ts_col] >= pd.Timestamp(start)]
    if end:
        df = df[df[ts_col] <= pd.Timestamp(end)]

    # filter mesin
    if machines:
        # normalisasi jadi uppercase
        _ms = [m.upper() for m in machines]
        df = df[df[machine_col].isin(_ms)]

    if df.empty:
        # kalau kosong, balikin df kosong tapi dengan kolom2 expected biar konsisten
        return df.assign(
            expected_min=np.nan,
            expected_max=np.nan,
            is_anomaly=False,
            which_bound=None,
            distance=np.nan,
            distance_pct=np.nan,
        )

    # build expected_min/expected_max row-wise
    exp_min_list = []
    exp_max_list = []

    # kita precompute distribusi per mesin (kalau fallback=infer)
    # supaya ga ngulang-ulang
    per_machine_stats: Dict[str, Tuple[float, float]] = {}
    if fallback == "infer":
        for mid, sub in df.groupby(machine_col):
            vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().astype(float)
            if len(vals) >= 10:
                vmin = float(vals.quantile(0.01))
                vmax = float(vals.quantile(0.99))
            else:
                # data terlalu sedikit -> pakai min/max actual
                vmin = float(vals.min()) if len(vals) else np.nan
                vmax = float(vals.max()) if len(vals) else np.nan
            per_machine_stats[mid] = (vmin, vmax)

    # loop baris
    for mid, val in zip(df[machine_col].astype(str), df[value_col].astype(float)):
        if mid in thresholds:
            vmin, vmax = thresholds[mid]
        else:
            if fallback == "skip":
                vmin, vmax = np.nan, np.nan
            else:
                vmin, vmax = per_machine_stats.get(mid, (np.nan, np.nan))
        exp_min_list.append(vmin)
        exp_max_list.append(vmax)

    df["expected_min"] = exp_min_list
    df["expected_max"] = exp_max_list

    # flag
    below_mask = df[value_col] < df["expected_min"]
    above_mask = df[value_col] > df["expected_max"]
    df["is_anomaly"] = below_mask | above_mask

    df["which_bound"] = np.where(
        below_mask,
        "below",
        np.where(above_mask, "above", None),
    )

    # distance absolut dari batas terdekat
    df["distance"] = np.where(
        df["which_bound"] == "below",
        df["expected_min"] - df[value_col],
        np.where(
            df["which_bound"] == "above",
            df[value_col] - df["expected_max"],
            0.0,
        ),
    )

    # distance relatif terhadap lebar rentang.
    # kalau lebar 0 / NaN → fallback ke |distance|
    width = (df["expected_max"] - df["expected_min"]).replace(0, np.nan)
    rel = np.where(
        df["is_anomaly"],
        df["distance"] / width,
        0.0,
    )
    rel = pd.Series(rel, index=df.index).fillna(df["distance"].abs())
    df["distance_pct"] = rel

    return df


# ------------------------------------------------------------
# 3) Ringkas per mesin
# ------------------------------------------------------------
def summarize_anomalies(
    df_anom: pd.DataFrame,
    *,
    ts_col: str = "ts",
    machine_col: str = "machine_id",
    value_col: str = "value",
    top_k: int = 5,
) -> dict:
    """
    Build summary per machine:
      total anomali
      earliest / latest anomaly timestamp
      worst_examples (titik paling parah berdasarkan distance_pct)
    Plus ringkasan gabungan.
    """
    if df_anom.empty:
        return {"machines": {}, "combined": {"total": 0}}

    res: Dict[str, dict] = {}

    for mid, g in df_anom.groupby(machine_col):
        total = int(g["is_anomaly"].sum())

        if total == 0:
            res[mid] = {"total": 0}
            continue

        g1 = g[g["is_anomaly"]].sort_values(ts_col)

        # earliest & latest anomaly ts
        earliest = g1[ts_col].iloc[0]
        latest = g1[ts_col].iloc[-1]

        # worst examples = anomaly dgn distance_pct terbesar
        worst_sel = (
            g1.sort_values("distance_pct", ascending=False)
              .head(top_k)[
                  [ts_col, value_col, "which_bound",
                   "expected_min", "expected_max",
                   "distance", "distance_pct"]
              ]
              .copy()
        )
        # cast ts ke string rapi
        worst_sel[ts_col] = pd.to_datetime(worst_sel[ts_col], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        res[mid] = {
            "total": total,
            "earliest": pd.to_datetime(earliest).isoformat(),
            "latest": pd.to_datetime(latest).isoformat(),
            "worst_examples": worst_sel.to_dict(orient="records"),
        }

    combined = {"total": int(df_anom["is_anomaly"].sum())}

    return {"machines": res, "combined": combined}


# ------------------------------------------------------------
# 4) Plot helper
# ------------------------------------------------------------
def plot_anomaly_series(
    df_anom: pd.DataFrame,
    machine_id: str,
    *,
    ts_col: str = "ts",
    value_col: str = "value",
    save_path: Optional[str] = None,
    title_suffix: str = "ANOMALY",
):
    """
    Plot 1 mesin:
    - garis nilai aktual
    - titik anomali (scatter)
    - garis batas min/max
    Note: kita tidak set warna custom (ikut policy matplotlib default).
    """
    d = df_anom[df_anom["machine_id"].astype(str).str.upper() == machine_id.upper()].copy()
    if d.empty:
        return None

    d = d.sort_values(ts_col)

    plt.figure()
    # garis nilai aktual
    plt.plot(d[ts_col], d[value_col], marker=".", linestyle="-")

    # titik anomali
    da = d[d["is_anomaly"]]
    if not da.empty:
        plt.scatter(da[ts_col], da[value_col], s=30)  # default color (policy: no custom color)

    # garis batas
    if d["expected_min"].notna().any():
        plt.plot(d[ts_col], d["expected_min"], linestyle="--")
    if d["expected_max"].notna().any():
        plt.plot(d[ts_col], d["expected_max"], linestyle="--")

    plt.xlabel("Waktu")
    plt.ylabel(value_col)
    plt.title(f"{machine_id} — {title_suffix}")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=140)
        plt.close()
        return save_path

    return None


# ------------------------------------------------------------
# 5) Orchestrator tinggi (pakai intent/subintent)
# ------------------------------------------------------------
def serve_anomaly_pipeline(
    df_raw: pd.DataFrame,
    thresholds_map: Dict[str, Tuple[float, float]],
    *,
    intent: Optional[dict] = None,
    subintent: Literal["anom_only", "count", "list", "plot", "summary"] = "summary",
    out_dir: str = "./plots_anom",
    ts_col: str = "ts",
    machine_col: str = "machine_id",
    value_col: str = "value",
) -> dict:
    """
    High-level wrapper yang:
      - baca intent (time_range.start/end, scope, machines)
      - panggil detect_range_anomalies()
      - summarize
      - optionally generate plot per mesin
      - return dict buat dikirim ke LLM / API

    intent format (opsional):
    {
        "time_range": { "start": "...", "end": "..." },
        "scope": "SUBSET"|"ALL",
        "machines": ["XP888A", "PQ667", ...]
    }
    """

    # 1) Extract filter dari intent
    start = end = None
    machines = None

    if intent:
        tr = intent.get("time_range") or {}
        start = tr.get("start")
        end = tr.get("end")

        if intent.get("scope") == "SUBSET" and intent.get("machines"):
            machines = intent["machines"]

    # 2) Deteksi anomali
    df_anom = detect_range_anomalies(
        df_raw,
        thresholds_map,
        ts_col=ts_col,
        machine_col=machine_col,
        value_col=value_col,
        start=start,
        end=end,
        machines=machines,
        fallback="infer",  # kamu bisa ganti "skip" buat strict
    )

    # 3) Ringkasan
    summary = summarize_anomalies(
        df_anom,
        ts_col=ts_col,
        machine_col=machine_col,
        value_col=value_col,
        top_k=5,
    )

    # 4) Plot (kalau subintent butuh visual)
    only_counts = subintent in {"count", "anom_only", "list"}
    plot_paths: Dict[str, str] = {}

    if not only_counts:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for mid in df_anom[machine_col].astype(str).str.upper().unique():
            save_p = os.path.join(out_dir, f"{mid}_anomaly.png")
            out_p = plot_anomaly_series(
                df_anom,
                machine_id=mid,
                ts_col=ts_col,
                value_col=value_col,
                save_path=save_p,
                title_suffix="ANOMALY vs RANGE",
            )
            if out_p:
                plot_paths[mid] = out_p

    # 5) Metadata ringkas
    meta = {
        "time_used": {"start": start, "end": end},
        "machines": machines or "ALL",
        "threshold_source": "sheet",
        "summary": summary,
    }

    return {
        "meta": meta,
        "plot_paths": plot_paths,
        "dataframe": df_anom,
    }


# ------------------------------------------------------------
# 6) Pretty text for human / LLM
# ------------------------------------------------------------
def pretty_anomaly_text(
    result: dict,
    *,
    decimal_comma: bool = True,
    places: int = 6,
) -> str:
    """
    Buat ringkasan teks manusia.
    Menampilkan total anomali global, lalu per mesin, plus worst_examples.
    """
    if result is None:
        return "No result."

    summary = result["meta"]["summary"]
    lines: List[str] = []

    def _fmt_num(v, p: int = 6) -> str:
        if v is None:
            return "NA"
        if isinstance(v, (float, int)):
            if np.isnan(v) or np.isinf(v):
                return "NA"
            s = f"{v:.{p}f}"
            return s.replace(".", ",") if decimal_comma else s
        return str(v)

    # total global
    lines.append(f"Total anomali: {summary['combined']['total']}.")

    # per mesin
    for mid, info in summary["machines"].items():
        total = info.get("total", 0)
        if total == 0:
            lines.append(f"- [{mid}] tidak ada anomali.")
            continue

        earliest = info.get("earliest", "")
        latest = info.get("latest", "")
        lines.append(
            f"- [{mid}] anomali={total}, "
            f"awal={earliest[:19]}, akhir={latest[:19]}."
        )

        # worst samples
        for w in info["worst_examples"]:
            ts_str = w.get("ts", "")[:19]
            which = w.get("which_bound", "")
            dist = _fmt_num(w.get("distance"), places)
            exp_min = _fmt_num(w.get("expected_min"), places)
            exp_max = _fmt_num(w.get("expected_max"), places)
            val_now = _fmt_num(w.get("value"), places)

            lines.append(
                f"    • {ts_str} | val={val_now} | {which} "
                f"({dist} dari batas; rentang [{exp_min}, {exp_max}])"
            )

    return "\n".join(lines)


# # ============================================================
# # PIPELINE ANOMALI: "NILAI DI LUAR RENTANG PER-MESIN"
# # ============================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import os
# from typing import Dict, Tuple, Optional, List

# # ---------- Utils ----------
# def _coerce_num(x):
#     if isinstance(x, str):
#         x = x.replace(",", ".")
#     return pd.to_numeric(x, errors="coerce")

# def _ensure_dir(p):
#     Path(p).parent.mkdir(parents=True, exist_ok=True)

# # ---------- 1) Load thresholds (min/max) per machine ----------
# def load_thresholds(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
#     """
#     Path bisa .xlsx/.xls/.csv. Sheet opsional untuk Excel.
#     Harus ada kolom: (case-insensitive, bebas spasi)
#       - tag number / machine / mesin
#       - maximum / max
#       - minimum / min
#     """
#     if path.lower().endswith((".xlsx", ".xls")):
#         df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
#     else:
#         df = pd.read_csv(path)
#     # normalisasi header
#     def norm(s): return (str(s) or "").strip().lower().replace(" ", "").replace("_","")
#     cols = {c: norm(c) for c in df.columns}
#     # map kolom
#     col_tag = None
#     for c,n in cols.items():
#         if n in {"tagnumber","tagnumber.","tag","machine","mesin"}:
#             col_tag = c; break
#     col_max = None
#     for c,n in cols.items():
#         if "max" in n: col_max = c; break
#     col_min = None
#     for c,n in cols.items():
#         if "min" in n: col_min = c; break
#     if not (col_tag and col_min and col_max):
#         raise ValueError(f"Kolom wajib tidak ditemukan. Ada: {list(df.columns)}")

#     out = df[[col_tag, col_max, col_min]].copy()
#     out.columns = ["machine_id", "max_value", "min_value"]
#     out["machine_id"] = out["machine_id"].astype(str).str.strip()
#     out["max_value"] = out["max_value"].apply(_coerce_num)
#     out["min_value"] = out["min_value"].apply(_coerce_num)

#     # bersihkan & swap kalau min > max
#     out = out.dropna(subset=["machine_id","max_value","min_value"]).reset_index(drop=True)
#     swap_mask = out["min_value"] > out["max_value"]
#     if swap_mask.any():
#         tmp = out.loc[swap_mask, "min_value"].copy()
#         out.loc[swap_mask, "min_value"] = out.loc[swap_mask, "max_value"]
#         out.loc[swap_mask, "max_value"] = tmp
#     return out

# def build_threshold_map(df_thresh: pd.DataFrame) -> Dict[str, Tuple[float,float]]:
#     return {row["machine_id"]: (float(row["min_value"]), float(row["max_value"])) for _,row in df_thresh.iterrows()}

# # ---------- 2) Detect anomalies (outside [min,max]) ----------
# def detect_range_anomalies(
#     df_raw: pd.DataFrame,
#     thresholds: Dict[str, Tuple[float,float]],
#     ts_col="ts", machine_col="machine_id", value_col="value",
#     start: Optional[str] = None, end: Optional[str] = None,
#     machines: Optional[List[str]] = None,
#     fallback="infer"   # "skip" → abaikan mesin tanpa threshold; "infer" → pakai quantile 1%-99%
# ) -> pd.DataFrame:
#     """
#     Return df dengan kolom tambahan:
#       - expected_min, expected_max
#       - is_anomaly (bool)
#       - which_bound ('below'/'above'/None)
#       - distance (nilai keluarannya)
#       - distance_pct (relatif terhadap rentang)
#     """
#     df = df_raw.copy()
#     df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
#     df[value_col] = df[value_col].apply(_coerce_num)
#     df = df.dropna(subset=[ts_col, machine_col, value_col])

#     if start: df = df[df[ts_col] >= pd.Timestamp(start)]
#     if end:   df = df[df[ts_col] <= pd.Timestamp(end)]
#     if machines:
#         df = df[df[machine_col].isin(machines)]

#     if df.empty:
#         return df.assign(expected_min=np.nan, expected_max=np.nan,
#                          is_anomaly=False, which_bound=None,
#                          distance=np.nan, distance_pct=np.nan)

#     # siapkan expected_min/expected_max per baris
#     mins = []; maxs = []
#     for mid,val in zip(df[machine_col].astype(str), df[value_col].astype(float)):
#         if mid in thresholds:
#             vmin, vmax = thresholds[mid]
#         else:
#             if fallback == "skip":
#                 vmin, vmax = np.nan, np.nan
#             else:
#                 # infer dari distribusi mesin itu di range terpilih
#                 s = df.loc[df[machine_col]==mid, value_col].astype(float)
#                 if len(s) >= 10:
#                     vmin, vmax = float(s.quantile(0.01)), float(s.quantile(0.99))
#                 else:
#                     vmin, vmax = float(s.min()), float(s.max())
#         mins.append(vmin); maxs.append(vmax)
#     df["expected_min"] = mins
#     df["expected_max"] = maxs

#     # flag anomaly
#     df["is_anomaly"] = (df[value_col] < df["expected_min"]) | (df[value_col] > df["expected_max"])
#     df["which_bound"] = np.where(df[value_col] < df["expected_min"], "below",
#                           np.where(df[value_col] > df["expected_max"], "above", None))
#     # distance absolut ke batas terdekat
#     df["distance"] = np.where(
#         df["which_bound"]=="below", df["expected_min"] - df[value_col],
#         np.where(df["which_bound"]=="above", df[value_col] - df["expected_max"], 0.0)
#     )
#     # distance relatif terhadap lebar rentang; kalau rentang 0 → pakai |val - bound|
#     width = (df["expected_max"] - df["expected_min"]).replace(0, np.nan)
#     df["distance_pct"] = np.where(
#         df["is_anomaly"],
#         df["distance"] / width.replace({0: np.nan}),
#         0.0
#     )
#     df["distance_pct"] = df["distance_pct"].fillna(df["distance"].abs())
#     return df

# # ---------- 3) Summaries per machine ----------
# def summarize_anomalies(df_anom: pd.DataFrame, ts_col="ts", machine_col="machine_id", value_col="value", top_k=5):
#     if df_anom.empty:
#         return {"machines": {}, "combined": {"total": 0}}
#     res = {}
#     for mid, g in df_anom.groupby(machine_col):
#         total = int(g["is_anomaly"].sum())
#         if total == 0:
#             res[mid] = {"total": 0}
#             continue
#         g1 = g[g["is_anomaly"]].sort_values(ts_col)
#         earliest = g1[ts_col].iloc[0].isoformat()
#         latest   = g1[ts_col].iloc[-1].isoformat()
#         # contoh anomali terparah (by distance_pct, descending)
#         worst = g1.sort_values("distance_pct", ascending=False).head(top_k)[
#             [ts_col, value_col, "which_bound", "expected_min", "expected_max", "distance", "distance_pct"]
#         ].copy()
#         worst[ts_col] = worst[ts_col].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
#         res[mid] = {
#             "total": total,
#             "earliest": earliest,
#             "latest": latest,
#             "worst_examples": worst.to_dict(orient="records")
#         }
#     combined = {"total": int(df_anom["is_anomaly"].sum())}
#     return {"machines": res, "combined": combined}

# # ---------- 4) Plot (highlight anomalies) ----------
# def plot_anomaly_series(df_anom, machine_id, *, ts_col="ts", value_col="value",
#                         save_path=None, title_suffix="ANOMALY"):
#     d = df_anom[df_anom["machine_id"]==machine_id].copy()
#     if d.empty: return None
#     d = d.sort_values(ts_col)
#     # garis normal
#     plt.figure()
#     plt.plot(d[ts_col], d[value_col], marker=".", linestyle="-")
#     # highlight anomaly
#     da = d[d["is_anomaly"]]
#     if not da.empty:
#         plt.scatter(da[ts_col], da[value_col], s=30)  # default color; policy: no custom colors
#     # overlay batas (opsional: step)
#     if d["expected_min"].notna().any():
#         plt.plot(d[ts_col], d["expected_min"], linestyle="--")
#     if d["expected_max"].notna().any():
#         plt.plot(d[ts_col], d["expected_max"], linestyle="--")
#     plt.xlabel("Waktu"); plt.ylabel(value_col); plt.title(f"{machine_id} — {title_suffix}")
#     plt.tight_layout()
#     if save_path:
#         _ensure_dir(save_path); plt.savefig(save_path, dpi=140); plt.close(); return save_path
#     return None

# # ---------- 5) Orkestrator: from intent + subintent ----------
# def serve_anomaly_pipeline(
#     df_raw: pd.DataFrame,
#     thresholds_map: Dict[str, Tuple[float,float]],
#     intent: Optional[dict] = None,     # boleh None → pakai semua data
#     subintent: str = "summary",        # "anom_only"|"count"|"list"|"plot"|"summary"
#     out_dir: str = "./plots_anom",
#     ts_col="ts", machine_col="machine_id", value_col="value"
# ):
#     # 1) filter waktu & mesin dari intent (kalau ada)
#     start = end = None; machines = None
#     if intent:
#         tr = intent.get("time_range") or {}
#         start, end = tr.get("start"), tr.get("end")
#         if (intent.get("scope") == "SUBSET") and intent.get("machines"):
#             machines = intent["machines"]

#     # 2) deteksi anomali
#     df_anom = detect_range_anomalies(
#         df_raw, thresholds_map, ts_col=ts_col, machine_col=machine_col, value_col=value_col,
#         start=start, end=end, machines=machines, fallback="infer"  # ubah ke "skip" kalau mau strict
#     )

#     # 3) ringkasan
#     summary = summarize_anomalies(df_anom, ts_col=ts_col, machine_col=machine_col, value_col=value_col, top_k=5)

#     # 4) efisiensi: kalau user cuma minta jumlah/list → skip plot
#     only_counts = subintent in {"count","anom_only","list"}
#     plot_paths = {}
#     if not only_counts:
#         Path(out_dir).mkdir(parents=True, exist_ok=True)
#         for mid in df_anom[machine_col].unique():
#             p = os.path.join(out_dir, f"{mid}_anomaly.png")
#             out = plot_anomaly_series(df_anom, mid, ts_col=ts_col, value_col=value_col,
#                                       save_path=p, title_suffix="ANOMALY vs RANGE")
#             if out:
#                 plot_paths[mid] = out

#     # 5) metadata ringkas untuk LLM
#     meta = {
#         "time_used": {"start": start, "end": end},
#         "machines": machines or "ALL",
#         "threshold_source": "sheet" ,
#         "summary": summary
#     }
#     return {"meta": meta, "plot_paths": plot_paths, "dataframe": df_anom}

# # ---------- 6) Pretty text ----------
# def pretty_anomaly_text(result, decimal_comma=True, places=6):
#     if result is None: return "No result."
#     s = result["meta"]["summary"]; lines = []
#     def fmt(x, p=6):
#         if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))): return "NA"
#         s = f"{x:.{p}f}"
#         return s.replace(".", ",") if decimal_comma else s
#     lines.append(f"Total anomali: {s['combined']['total']}.")
#     for mid, v in s["machines"].items():
#         if v.get("total",0) == 0:
#             lines.append(f"- [{mid}] tidak ada anomali.")
#             continue
#         lines.append(f"- [{mid}] anomali={v['total']}, awal={v['earliest'][:19]}, akhir={v['latest'][:19]}.")
#         for w in v["worst_examples"]:
#             lines.append(
#                 f"    • {w['ts']} | val={fmt(w['value'],places)} | {w['which_bound']} "
#                 f"({fmt(w['distance'],places)} dari batas; rentang [{fmt(w['expected_min'],places)}, {fmt(w['expected_max'],places)}])"
#             )
#     return "\n".join(lines)
