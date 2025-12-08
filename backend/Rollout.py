# rollout.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .forecaster import HORIZON_MINUTES, HORIZON_BY_MIN

def shift_lags(state_lags: Dict[str, float], new_value: float) -> Dict[str, float]:
    """
    Geser jendela lag sesudah kita memproyeksikan satu langkah ke depan.
    Asumsi:
      y_lag_1 = nilai paling baru
      y_lag_2 = nilai sebelumnya
      y_lag_3 = nilai sebelumnya lagi
    Setelah prediksi baru keluar, prediksi tsb jadi y_lag_1 baru.
    """
    new_state = state_lags.copy()
    if "y_lag_3" in new_state:
        new_state["y_lag_3"] = state_lags["y_lag_2"]
    if "y_lag_2" in new_state:
        new_state["y_lag_2"] = state_lags["y_lag_1"]
    new_state["y_lag_1"] = float(new_value)
    return new_state


def _predict_hop_minutes(forecaster, state_lags: Dict[str, float], hop_min: int) -> float:
    """
    Prediksi untuk satu hop ke depan sejauh hop_min menit.
    hop_min harus cocok dengan salah satu horizon head (3,6,12,24,48).
    """
    horizon_key = HORIZON_BY_MIN[hop_min]
    return forecaster.predict_single_horizon(horizon_key, state_lags)


def _choose_hop_sequence_cover(target_minutes: int, already_covered: int) -> List[int]:
    """
    Buat daftar hop tambahan supaya kita mencapai (atau melebihi dikit)
    target_minutes dari posisi waktu saat ini (already_covered).
    Ini dipakai untuk multi target (multi horizon request).
    """
    hops: List[int] = []
    t = already_covered

    while t < target_minutes:
        remaining = target_minutes - t

        # heuristik adaptif
        if remaining <= 12:
            candidates = [3, 6, 12, 24, 48]
        else:
            candidates = [48, 24, 12, 6, 3]

        chosen = None
        # pilih kandidat paling kecil yang masih >= remaining
        for c in candidates:
            if c >= remaining:
                chosen = c
                break

        if chosen is None:
            # gak ada satu hop yang >= remaining;
            # pilih hop "besar" biar cepat maju
            chosen = candidates[-1] if remaining <= 12 else candidates[0]

        hops.append(chosen)
        t += chosen

    return hops


def choose_hop_sequence(target_minutes: int) -> List[int]:
    """
    Heuristik hop sequence untuk 1 target akhir.
    Contoh:
      - <=6   -> [3,3] atau [3]
      - <=12  -> [12]
      - <=24  -> [12,12] (kecuali pas 24 → [24])
      - <=48  -> [24] / [48] tergantung jarak
      - <=60  -> [12,48]
      - >60   -> pattern blok [12,48] berulang lalu sisa dipecah
    Ini adalah policy domain (boleh di-tune kemudian).
    """
    t = int(target_minutes)

    if t <= 6:
        return [3, 3] if t > 3 else [3]

    if t <= 12:
        return [12]

    if t <= 24:
        return [12, 12] if t < 24 else [24]

    if t <= 48:
        if t <= 30:
            return [24]
        else:
            # kamu kadang pakai [48] untuk >30 dan <=48 di versi awal.
            return [48]

    if t <= 60:
        return [12, 48]

    # t > 60 -> ulang blok 60 menit ([12,48]) sebanyak mungkin
    hops: List[int] = []
    remaining = t
    while remaining > 60:
        hops.extend([12, 48])  # total 60
        remaining -= 60

    # handle sisa <=60
    if remaining > 48:
        hops.append(48)
        remaining -= 48
    elif remaining > 24:
        hops.append(24)
        remaining -= 24
    elif remaining > 12:
        hops.append(12)
        remaining -= 12

    if remaining > 6:
        hops.append(6)
        remaining -= 6
    if remaining > 0:
        hops.append(3)

    return hops


def rollout_autoregressive(
    forecaster,
    init_state_lags: Dict[str, float],
    target_minutes: int,
) -> Dict[str, Any]:
    """
    Simulasi autoregressive sampai ~target_minutes di masa depan.
    Setiap hop:
      - prediksi nilai
      - tambahkan lamanya hop ke t_accum
      - update lag state jadi nilai prediksi
    Return dict:
    {
      "trajectory": [
        {"t_accum": 12, "hop": 12, "pred": 105.3},
        {"t_accum": 60, "hop": 48, "pred": 106.1},
        ...
      ],
      "final_pred": 106.1,
      "t_final": 60
    }
    """
    hops = choose_hop_sequence(target_minutes)
    traj: List[Dict[str, Any]] = []
    state = init_state_lags.copy()
    t_accum = 0

    for hop_min in hops:
        y_pred = _predict_hop_minutes(forecaster, state, hop_min)
        t_accum += hop_min
        traj.append({
            "t_accum": t_accum,
            "hop": hop_min,
            "pred": y_pred,
        })
        state = shift_lags(state, y_pred)

    return {
        "trajectory": traj,
        "final_pred": traj[-1]["pred"],
        "t_final": traj[-1]["t_accum"],
    }


def rollout_incremental_horizons(
    forecaster,
    init_state_lags: Dict[str, float],
    targets_minutes_sorted: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Jalankan simulasi maju sekali terus, tapi simpan snapshot
    di setiap target horizon yang diminta user.
    Misal user bilang "12, 24, 48 menit": kita gak mau reset ulang dari awal
    tiap target; kita terusin chain yang sama.
    Return:
      {
        12: {
          "pred_value": ...,
          "t_effective": 12,
          "trajectory_so_far": [...]
        },
        24: {...},
        ...
      }
    """
    results: Dict[int, Dict[str, Any]] = {}
    state = init_state_lags.copy()
    t_accum = 0
    traj_global: List[Dict[str, Any]] = []

    for tgt in targets_minutes_sorted:
        extra_hops = _choose_hop_sequence_cover(tgt, t_accum)

        for hop_min in extra_hops:
            y_pred = _predict_hop_minutes(forecaster, state, hop_min)
            t_accum += hop_min
            traj_global.append({
                "t_accum": t_accum,
                "hop": hop_min,
                "pred": y_pred,
            })
            state = shift_lags(state, y_pred)

        results[tgt] = {
            "pred_value": traj_global[-1]["pred"],
            "t_effective": t_accum,
            "trajectory_so_far": list(traj_global),
        }

    return results


def build_range_targets(start_min: int, end_min: int) -> List[int]:
    """
    Untuk permintaan RANGE (misal 0..30 menit), kita perlu grid titik
    untuk dievaluasi.
    Strategy sederhana:
      - ambil horizon default [3,6,12,24,48]
      - keep yang jatuh di antara start_min..end_min
      - pastikan end_min juga ikut
    """
    base_pts = sorted(HORIZON_MINUTES.values())  # [3,6,12,24,48]
    pts = [m for m in base_pts if (m >= start_min and m <= end_min)]
    if end_min not in pts:
        pts.append(end_min)
    pts = sorted(set(pts))
    return pts
