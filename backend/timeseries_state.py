# timeseries_state.py
from __future__ import annotations

from typing import Tuple, Dict, List, Optional
from datetime import datetime
import pandas as pd


def load_raw_timeseries(csv_path: str = "/workspace/df_raw2.csv") -> pd.DataFrame:
    """
    Load CSV sensor mentah.
    Wajib ada kolom:
        - machine_id
        - ts               (timestamp string / ISO-ish)
        - value            (float / suhu / metric utama yang diprediksi)
    Return dataframe yang sudah:
        - ts -> datetime64[ns]
        - sorted per (machine_id, ts)
        - reset_index
    """
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"])
    df = (
        df.sort_values(["machine_id", "ts"])
          .reset_index(drop=True)
    )
    return df


def get_all_machine_ids(df_raw: pd.DataFrame) -> List[str]:
    """
    Ambil daftar machine_id unik yang ada di df_raw dalam bentuk sorted list.
    """
    return sorted(df_raw["machine_id"].unique().tolist())


def build_initial_lag_state(
    df_raw: pd.DataFrame,
    machine_id: str,
    anchor_time: Optional[datetime] = None,
    n_lags: int = 3,
) -> Tuple[Dict[str, float], datetime]:
    """
    Ambil kondisi 'state lag' terakhir untuk satu mesin.

    df_raw:
        dataframe full time series semua mesin
    machine_id:
        mesin spesifik yang mau dievaluasi
    anchor_time:
        - None  -> ambil row terbaru (ts paling akhir) untuk mesin tsb
        - datetime -> ambil row terakhir dengan ts <= anchor_time
    n_lags:
        berapa lag yang mau diambil (misal 3 -> y_lag_1, y_lag_2, y_lag_3)

    Returns
    -------
    (lag_state, anchor_used_ts)
    lag_state: dict seperti
        {
          "y_lag_1": 105.2,
          "y_lag_2": 104.8,
          "y_lag_3": 104.5
        }
      y_lag_1 = nilai termutakhir,
      y_lag_2 = sebelumnya, dst.
      Kalau datanya kurang panjang, nilai terakhir di-duplicate untuk padding.
    anchor_used_ts: timestamp (pd.Timestamp) yang akhirnya dipakai.
    """
    df_m = df_raw[df_raw["machine_id"] == machine_id].copy()
    if df_m.empty:
        raise RuntimeError(f"no data for machine {machine_id}")

    # filter sampai anchor_time kalau diminta
    if anchor_time is not None:
        df_m = df_m[df_m["ts"] <= anchor_time]

    # urutkan descending biar baris 0 = terbaru
    df_m = (
        df_m.sort_values("ts", ascending=False)
            .reset_index(drop=True)
    )

    if len(df_m) < 1:
        raise RuntimeError(
            f"machine {machine_id}: no rows before anchor_time {anchor_time}"
        )

    # ambil kolom value jadi list float terbaru → lama
    vals = df_m["value"].astype(float).tolist()

    # padding kalau data kurang panjang
    while len(vals) < n_lags:
        vals.append(vals[-1])
    vals = vals[:n_lags]

    lag_state = {
        f"y_lag_{i+1}": float(v)
        for i, v in enumerate(vals)
    }

    anchor_used_ts = df_m.loc[0, "ts"]
    return lag_state, anchor_used_ts
