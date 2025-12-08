# rollout.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

# Mapping horizon model → menit.
# forecaster.predict_single_horizon() pakai key seperti "y_step_12m"
HORIZON_MINUTES = {
    "y_step_3m": 3,
    "y_step_6m": 6,
    "y_step_12m": 12,
    "y_step_24m": 24,
    "y_step_48m": 48,
}

# Balikan: dari menit → nama head model
HORIZON_BY_MIN = {v: k for k, v in HORIZON_MINUTES.items()}


def shift_lags(state_lags: Dict[str, float], new_value: float) -> Dict[str, float]:
    """
    Geser window lag setelah kita 'melompat' ke masa depan.

    Asumsi struktur state_lags:
        {
          "y_lag_1": <nilai terbaru>,
          "y_lag_2": <nilai sebelumnya>,
          "y_lag_3": <nilai lebih lama>,
          ...
        }

    Setelah prediksi nilai masa depan (new_value), nilai itu jadi y_lag_1 baru.
    y_lag_1 lama -> y_lag_2, y_lag_2 lama -> y_lag_3, dst.

    Catatan: sekarang kita hardcode 3 lag. Kalau nanti mau 5 lag,
    tinggal diperluas logic ini.
    """
    new_state = state_lags.copy()
    new_state["y_lag_3"] = state_lags["y_lag_2"]
    new_state["y_lag_2"] = state_lags["y_lag_1"]
    new_state["y_lag_1"] = float(new_value)
    return new_state


def onestep_predict_minutes(
    forecaster,
    state_lags: Dict[str, float],
    horizon_min: int
) -> float:
    """
    Minta 1 prediksi value pada horizon tertentu (misal 12 menit ahead)
    dari kondisi 'state_lags' saat ini.

    forecaster: HorizonForecasterClassic (atau kompatibel) yang punya:
        predict_single_horizon(horizon_key: str, state_dict: dict) -> float

    horizon_min: jumlah menit yang ingin kita lompati dalam satu hop.
                 contoh 12, 24, 48.

    return: pred_value (float)
    """
    # Cari nama head model berdasarkan menit.
    # Misal 12 → "y_step_12m"
    hk = HORIZON_BY_MIN[horizon_min]
    pred_val = forecaster.predict_single_horizon(hk, state_lags)
    return pred_val


def choose_hop_sequence(target_minutes: int) -> List[int]:
    """
    Tentukan urutan lompatan horizon (hop) untuk mendekati target_minutes.

    Kenapa perlu hop sequence?
    - Model klasik kamu kelihatannya dilatih untuk horizon diskrit
      (3m, 6m, 12m, 24m, 48m).  :contentReference[oaicite:2]{index=2}
    - Untuk memprediksi '120 menit ke depan' kita gak punya head 120m langsung,
      jadi kita rangkai beberapa hop (misal 12,48,48,12) supaya akumulatif ~120.

    Heuristik di sini berasal dari kode awal kamu:
    - <=6  -> [3,3] (atau [6])
    - <=12 -> [12]
    - <=24 -> [12,12] atau [24]
    - <=48 -> [24] atau [24,24]
    - <=60 -> [12,48]
    - >60  -> ulang pola [12,48] beberapa kali, lalu pecah sisa.

    Ini bukan ilmu pasti; ini "policy" rollout. Boleh di-tune nanti.
    """

    t = int(target_minutes)

    if t <= 6:
        return [3, 3] if t > 3 else [3]

    if t <= 12:
        return [12]

    if t <= 24:
        # kalau kurang dari 24 tapi >12 kita pakai dua lompatan 12
        # kalau persis 24 kita boleh pakai 24 langsung
        return [12, 12] if t < 24 else [24]

    if t <= 48:
        # kalau dekat 30-an menit: mungkin [24] cukup untuk "mendekati"
        # kalau mendekati 48 penuh: [24,24]
        if t <= 30:
            return [24]
        elif t > 30:
            return [24, 24]
        else:
            return [24]

    if t <= 60:
        return [12, 48]

    # t > 60
    hops: List[int] = []
    remaining = t

    # ambil blok 60 menit dengan [12,48]
    while remaining > 60:
        hops.extend([12, 48])
        remaining -= 60

    # Sisa <= 60
    if remaining > 48:
        hops.append(48)
        remaining -= 48
    elif remaining > 24:
        hops.append(24)
        remaining -= 24
    elif remaining > 12:
        hops.append(12)
        remaining -= 12

    # kalau masih ada sisa kecil (<12), tambahkan 6/3 dsb
    if remaining > 6:
        hops.append(6)
        remaining -= 6
    if remaining > 0:
        hops.append(3)

    return hops


def rollout_autoregressive(
    forecaster,
    init_state_lags: Dict[str, float],
    target_minutes: int
) -> Dict[str, Any]:
    """
    Lakukan simulasi autoregressive ke depan sampai mendekati target_minutes
    dengan urutan hop dari choose_hop_sequence().

    - forecaster: objek model forecasting klasik kamu.
                  Harus punya method predict_single_horizon(hk, state_dict).
    - init_state_lags: dict lag awal pada timestamp anchor sekarang.
    - target_minutes: misal 120 → kita "jalan" sampai ~120 menit ke depan.

    return:
    {
        "trajectory": [
            {"t_accum": 12, "pred": v12, "hop": 12},
            {"t_accum": 60, "pred": v60, "hop": 48},
            ...
        ],
        "final_pred": v_last,
        "t_final": t_last
    }
    """

    hops = choose_hop_sequence(target_minutes)
    traj = []
    state = init_state_lags.copy()
    t_accum = 0

    for hop_min in hops:
        # prediksi nilai setelah hop_min menit dari state saat ini
        pred_val = onestep_predict_minutes(forecaster, state, hop_min)

        # akumulasi waktu
        t_accum += hop_min

        # simpan titik di trajectory
        traj.append({
            "t_accum": t_accum,
            "pred": pred_val,
            "hop": hop_min
        })

        # update state_lags:
        # forecast di t_accum sekarang kita anggap jadi "nilai terbaru"
        state = shift_lags(state, pred_val)

    final_pred = traj[-1]["pred"]
    t_final    = traj[-1]["t_accum"]

    return {
        "trajectory": traj,
        "final_pred": final_pred,
        "t_final": t_final,
    }


def build_range_targets(start_min: int, end_min: int) -> List[int]:
    """
    Helper untuk RANGE request.

    Misal user minta "0 sampai 120 menit ke depan",
    kita gak mau nge-simulate setiap menit (mahal).
    Biasanya kita sampling grid kasar yang meaningful.

    Strategy simple:
    - selalu include start_min & end_min
    - juga include breakpoint 'menit penting' berbasis head model:
      3,6,12,24,48, ... selama masih di dalam range.

    Output harus sorted unik.
    Ini dipakai di execute_spec() buat nge-loop rollout_autoregressive()
    untuk tiap target tmin. :contentReference[oaicite:3]{index=3}
    """
    base_points = sorted(HORIZON_BY_MIN.keys())  # [3,6,12,24,48,...]
    pts = [start_min, end_min]

    for m in base_points:
        if m >= start_min and m <= end_min:
            pts.append(m)

    # buang nilai negatif / duplikat, urutkan
    cleaned = sorted({p for p in pts if p is not None and p >= 0})
    return cleaned


# import numpy as np

# HORIZON_MINUTES = {
#     "y_step_3m": 3,
#     "y_step_6m": 6,
#     "y_step_12m": 12,
#     "y_step_24m": 24,
#     "y_step_48m": 48,
# }

# # kebalikan biar gampang panggil model dari menit:
# HORIZON_BY_MIN = {v:k for k,v in HORIZON_MINUTES.items()}


# def shift_lags(state_lags, new_value):
#     """
#     Geser lag window setelah kita 'bergerak' ke masa depan.
#     Asumsi lag_1 = nilai paling baru, lag_2 = sebelumnya, dst.
#     """
#     new_state = state_lags.copy()
#     # contoh kita cuma punya 3 lag
#     new_state["y_lag_3"] = state_lags["y_lag_2"]
#     new_state["y_lag_2"] = state_lags["y_lag_1"]
#     new_state["y_lag_1"] = float(new_value)
#     return new_state


# def onestep_predict_minutes(forecaster, state_lags, horizon_min):
#     """
#     Pakai model horizon tertentu (misal 12 menit) untuk prediksi value masa depan relatif dari state_lags sekarang.
#     Return pred_value (float).
#     """
#     # model key
#     hk = HORIZON_BY_MIN[horizon_min]
#     pred_val = forecaster.predict_single_horizon(hk, state_lags)
#     return pred_val


# def choose_hop_sequence(target_minutes):
#     """
#     Ini adalah tempat kita encode strategi 'pakai 12 step lalu 48 step lalu ...'
#     Kamu bisa tweak sesuai kebijakan produksi.
#     Beberapa contoh rule (heuristic):
#     - <=6  -> [3,3]  (atau [6] kalo mau langsung)
#     - <=12 -> [12]
#     - <=24 -> [12,12] atau [24]
#     - <=48 -> [24,24] atau [48]
#     - <=60 -> [12,48]
#     - >60  -> [12,48,48,12]   # contoh pattern kamu untuk ~120 menit
#     Catatan: hasil total hop kita tidak harus persis target, tapi mendekati.
#     Nanti kita hitung menit kumulatif biar keliatan sampai berapa menit sebenernya.
#     """
#     t = target_minutes
#     if t <= 6:
#         return [3,3] if t > 3 else [3]
#     if t <= 12:
#         return [12]
#     if t <= 24:
#         return [12,12] if t < 24 else [24]
#     if t <= 48:
#         return [24] if t <= 30 else [24,24] if t > 30 else [24]
#     if t <= 60:
#         return [12,48]
#     # t > 60
#     # pattern kamu buat 2 jam (120) adalah 12,48,48,12
#     # kalau t super besar, kita bisa ulangin pola ini
#     # simple approach:
#     hops = []
#     remaining = t
#     while remaining > 60:
#         hops.extend([12,48])  # dua langkah, total 60
#         remaining -= 60
#     # sisa 'remaining' sekarang <=60
#     if remaining > 48:
#         hops.append(48); remaining -= 48
#     elif remaining > 24:
#         hops.append(24); remaining -= 24
#     elif remaining > 12:
#         hops.append(12); remaining -= 12
#     # kalau masih ada sisa kecil (<12), tambahin 3/6/12 kecil
#     if remaining > 6:
#         hops.append(6); remaining -= 6
#     if remaining > 0:
#         hops.append(3)
#     return hops


# def rollout_autoregressive(forecaster, init_state_lags, target_minutes):
#     """
#     Jalankan hop sequence sampai mendekati target_minutes.
#     - forecaster: HorizonForecasterClassic
#     - init_state_lags: dict {"y_lag_1":..., "y_lag_2":..., "y_lag_3":...}
#     - target_minutes: misal 120
#     Return:
#       {
#         "trajectory": [
#             {"t_accum": 12, "pred": v12},
#             {"t_accum": 60, "pred": v60},
#             {"t_accum": 108, "pred": v108},
#             {"t_accum": 120, "pred": v120}
#         ],
#         "final_pred": v_last,
#         "t_final": t_last
#       }
#     """

#     hops = choose_hop_sequence(target_minutes)
#     traj = []
#     state = init_state_lags.copy()
#     t_accum = 0

#     for hop_min in hops:
#         pred_val = onestep_predict_minutes(forecaster, state, hop_min)
#         t_accum += hop_min
#         traj.append({"t_accum": t_accum, "pred": pred_val, "hop": hop_min})
#         # update lags untuk langkah berikutnya
#         state = shift_lags(state, pred_val)

#     # final value = pred setelah hop terakhir
#     final_pred = traj[-1]["pred"]
#     t_final    = traj[-1]["t_accum"]

#     return {
#         "trajectory": traj,
#         "final_pred": final_pred,
#         "t_final": t_final
#     }
