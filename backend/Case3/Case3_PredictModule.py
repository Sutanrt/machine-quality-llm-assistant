############################################################
# 0. IMPORT & PREP
############################################################

import re, json, numpy as np
from pathlib import Path

# === asumsi ini sudah ada dari step sebelumnya ===
# from your_nlu_module import ForecastMultiRunner, interpret_query
# from your_training_module import HorizonForecasterClassic
# plus torch / joblib dll sudah diimport

############################################################
# 1. HORIZON FORECASTER (classical Linear+RF residual)
############################################################

import joblib

HORIZON_MINUTES = {
    "y_step_3m": 3,
    "y_step_6m": 6,
    "y_step_12m": 12,
    "y_step_24m": 24,
    "y_step_48m": 48,
}
HORIZON_BY_MIN = {v:k for k,v in HORIZON_MINUTES.items()}

class HorizonForecasterClassic:
    """
    Loader + predictor untuk model classical combo (Linear + RF residual)
    yang kamu simpan per horizon di outputs_<timestamp>/<horizon>/
    """
    def __init__(self, run_root):
        self.run_root = Path(run_root)
        self.cache = {}  # { "y_step_12m": {lin, rf, use_lags} }

    def _load_horizon(self, horizon_key):
        if horizon_key in self.cache:
            return self.cache[horizon_key]

        hz_dir = self.run_root / horizon_key
        lin = joblib.load(hz_dir / f"{horizon_key}_linear.joblib")
        rf  = joblib.load(hz_dir / f"{horizon_key}_rf_residual.joblib")

        cfg = json.loads((hz_dir / f"{horizon_key}_config.json").read_text())
        use_lags = cfg["use_lags"]  # contoh: ["y_lag_1","y_lag_2","y_lag_3"]

        self.cache[horizon_key] = {
            "lin": lin,
            "rf": rf,
            "use_lags": use_lags,
        }
        return self.cache[horizon_key]

    def predict_single_horizon(self, horizon_key, state_lags):
        """
        horizon_key: 'y_step_12m' dll
        state_lags: dict fitur realtime saat ini:
            {
                "y_lag_1": <float latest value>,
                "y_lag_2": <float prev>,
                "y_lag_3": <float prevprev>,
                ...
            }
        return float pred_combo
        """
        bundle = self._load_horizon(horizon_key)
        use_lags = bundle["use_lags"]

        X_now = np.array([[ state_lags[feat] for feat in use_lags ]], dtype=float)

        y_lin = bundle["lin"].predict(X_now)[0]
        y_res = bundle["rf"].predict(X_now)[0]
        y_combo = float(y_lin + y_res)
        return y_combo


############################################################
# 2. AUTOREGRESSIVE ROLLOUT ENGINE
############################################################

def shift_lags(state_lags, new_value):
    """
    Geser lag window setelah kita melangkah ke masa depan.
    Asumsi:
    - y_lag_1 = nilai terkini
    - y_lag_2 = nilai sebelumnya
    - y_lag_3 = nilai sebelumnya lagi
    Kita inject prediksi baru sebagai y_lag_1 berikutnya.
    """
    new_state = state_lags.copy()
    # geser ke belakang
    if "y_lag_3" in new_state:
        new_state["y_lag_3"] = state_lags["y_lag_2"]
    if "y_lag_2" in new_state:
        new_state["y_lag_2"] = state_lags["y_lag_1"]
    # y_lag_1 update dengan prediksi baru
    new_state["y_lag_1"] = float(new_value)
    return new_state

def onestep_predict_minutes(forecaster, state_lags, horizon_min):
    """
    Pakai model horizon_min (3,6,12,24,48) untuk prediksi nilai future.
    """
    hk = HORIZON_BY_MIN[horizon_min]
    return forecaster.predict_single_horizon(hk, state_lags)

 
def choose_hop_sequence_cover(target_minutes, already_covered):
    """
    Balikin list hops baru untuk maju dari already_covered
    sampai >= target_minutes.
    Tidak boleh berhenti di bawah target.
    """
    hops = []
    t = already_covered

    while t < target_minutes:
        remaining = target_minutes - t

        # mode adaptif
        if remaining <= 12:
            candidates = [3,6,12,24,48]
        else:
            candidates = [48,24,12,6,3]

        chosen = None
        # pilih kandidat terkecil yang masih >= remaining
        for c in candidates:
            if c >= remaining:
                chosen = c
                break

        if chosen is None:
            # tidak ada single hop yang >= remaining,
            # ambil kandidat terbesar biar cepet maju
            chosen = candidates[-1] if remaining <= 12 else candidates[0]

        hops.append(chosen)
        t += chosen

    return hops


def choose_hop_sequence(target_minutes):
    """
    Heuristik sequence hop untuk autoregressive chain.
    Bisa (dan harus) kamu tweak sesuai domain.
    - <=6  -> [3,3] (kasih dua hop 3 menit)
    - <=12 -> [12]
    - <=24 -> [12,12] kalau <24, atau [24] kalau pas 24
    - <=48 -> [24,24] kalau >24, else [24]
    - <=60 -> [12,48]
    - >60  -> pecah jadi blok 60 menit pakai [12,48] berulang,
              lalu sisa terakhir dihajar pakai potongan yang mendekati.
    Contoh 120 menit → [12,48,48,12] ~ 120m total.
    """
    t = target_minutes
    if t <= 6:
        return [3,3] if t > 3 else [3]
    if t <= 12:
        return [12]
    if t <= 24:
        if t < 24:
            return [12,12]
        else:
            return [24]
    if t <= 48:
        if t <= 30:
            return [24]
        else:
            return [48]
    if t <= 60:
        return [12,48]

    # t > 60 → kita terus kurangi 60 pakai blok [12,48]
    hops = []
    remaining = t
    while remaining > 60:
        hops.extend([12,48])  # ini total 60 per loop
        remaining -= 60
    # handle sisa <=60
    if remaining > 48:
        hops.append(48); remaining -= 48
    elif remaining > 24:
        hops.append(24); remaining -= 24
    elif remaining > 12:
        hops.append(12); remaining -= 12
    # kalau masih ada sisa kecil, tambahin 6/3
    if remaining > 6:
        hops.append(6); remaining -= 6
    if remaining > 0:
        hops.append(3)
    return hops
def predict_hop_minutes(forecaster, state_lags, hop_min):
    """
    1 hop = prediksi ke depan sejauh hop_min menit pakai horizon model terdekat.
    hop_min HARUS salah satu [3,6,12,24,48].
    """
    horizon_key = HORIZON_BY_MIN[hop_min]
    return forecaster.predict_single_horizon(horizon_key, state_lags)
    
def rollout_incremental_horizons(forecaster, init_state_lags, targets_minutes_sorted):
    """
    targets_minutes_sorted: ex [12,24,48] ATAU [3,24,50,120]
    Jalan maju stepwise, simpan snapshot untuk tiap target.
    """
    results = {}
    state = init_state_lags.copy()
    t_accum = 0
    traj_global = []

    for tgt in targets_minutes_sorted:
        extra_hops = choose_hop_sequence_cover(tgt, t_accum)

        for hop_min in extra_hops:
            y_pred = predict_hop_minutes(forecaster, state, hop_min)
            t_accum += hop_min
            traj_global.append({
                "t_accum": t_accum,
                "hop": hop_min,
                "pred": y_pred
            })
            state = shift_lags(state, y_pred)

        # simpan hasil setelah kita minimal sampai target ini
        results[tgt] = {
            "pred_value": traj_global[-1]["pred"],
            "t_effective": t_accum,
            "trajectory_so_far": list(traj_global),  # copy for debug/audit
        }

    return results

    
def rollout_autoregressive(forecaster, init_state_lags, target_minutes):
    """
    Simulasi maju step-by-step sesuai hop sequence.
    Setelah tiap hop:
      - prediksi nilai masa depan
      - update lag dengan nilai prediksi tsb
    Output:
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
        val_pred = onestep_predict_minutes(forecaster, state, hop_min)
        t_accum += hop_min
        traj.append({
            "t_accum": t_accum,
            "hop": hop_min,
            "pred": val_pred
        })
        # update kondisi jadi prediksi terakhir
        state = shift_lags(state, val_pred)

    return {
        "trajectory": traj,
        "final_pred": traj[-1]["pred"],
        "t_final": traj[-1]["t_accum"]
    }

def build_range_targets(start_min, end_min):
    """
    Untuk kasus RANGE (misal 0..30 menit ke depan),
    kita mau beberapa titik penting.
    Strategi simple:
      - gunakan horizon default [3,6,12,24,48]
      - ambil yang jatuh di window
      - pastikan titik akhir end_min juga ada
    """
    base_pts = sorted(HORIZON_MINUTES.values())  # [3,6,12,24,48]
    pts = [m for m in base_pts if (m >= start_min and m <= end_min)]
    if end_min not in pts:
        pts.append(end_min)
    pts = sorted(list(set(pts)))
    return pts

def forecast_point_autoreg(spec, forecaster, context_state):
    """
    Untuk SINGLE_POINT (contoh: '2 jam lagi berapa?').
    """
    target_min = spec["time"]["window_minutes"]["end_min"]  # sudah disnap jadi 120 dll
    rollout = rollout_autoregressive(forecaster, context_state, target_min)
    return {
        "target_request_min": target_min,
        "final_pred": rollout["final_pred"],
        "t_final": rollout["t_final"],
        "trajectory": rollout["trajectory"]
    }

def forecast_range_autoreg(spec, forecaster, context_state):
    """
    Untuk RANGE_REL/RANGE_FUTURE (contoh: '30 menit ke depan, interval 3 menit').
    Kita generate beberapa titik dalam window, dan jalankan rollout_autoregressive terpisah untuk tiap titik.
    """
    w = spec["time"]["window_minutes"]
    start_m = w["start_min"]
    end_m   = w["end_min"]

    targets = build_range_targets(start_m, end_m)
    series = []
    for tmin in targets:
        r = rollout_autoregressive(forecaster, context_state, tmin)
        series.append({
            "minute_ahead": tmin,
            "pred_value": r["final_pred"],
            "t_final": r["t_final"],
            "trajectory": r["trajectory"]
        })
    return {
        "window": [start_m, end_m],
        "timeline": series
    }

############################################################
# 3. EXECUTOR UNTUK SPEC (gabung ke pipeline)
############################################################

def execute_spec(spec, forecaster, context_state):
    """
    spec         : hasil interpret_query
    forecaster   : HorizonForecasterClassic instance
    context_state: dict lag realtime untuk mesin tsb
                   ex {"y_lag_1":105.2,"y_lag_2":104.8,"y_lag_3":104.5}
    return       : dict hasil final inference siap dikonsumsi UI / downstream
    """

    action = spec["action_type"]
    tinfo  = spec["time"]
    tc     = tinfo["time_complexity"]

   
    time_info = spec["time"]
    multi_list = time_info.get("multi_targets_minutes")

    # 0. MULTI-HORIZON QUERY (misal "12, 24, 48 menit")
    if multi_list and len(multi_list) > 1:
        # pastikan ascending
        reqs = sorted(multi_list)

        mh_res = rollout_incremental_horizons(
            forecaster,
            context_state,
            reqs
        )

        return {
            "type": "forecast_multi_point",
            "machines": spec["machines"],
            "requested_minutes": reqs,
            "per_horizon": {
                str(t): {
                    "pred_value": mh_res[t]["pred_value"],
                    "t_effective": mh_res[t]["t_effective"]
                }
                for t in reqs
            }
        }
    # counterfactual / forecast_numeric / forecast_trend -> numeric forecast forward
    if action in ["forecast_numeric", "forecast_trend", "counterfactual_forecast"]:
        if tc == "SINGLE_POINT":
            out_single = forecast_point_autoreg(spec, forecaster, context_state)
            return {
                "type": action,
                "machines": spec["machines"],
                "mode": "autoregressive_chain",
                "result_single": out_single,
                "counterfactual": spec["counterfactual"] if action=="counterfactual_forecast" else None
            }
        else:
            out_range = forecast_range_autoreg(spec, forecaster, context_state)
            return {
                "type": action,
                "machines": spec["machines"],
                "mode": "autoregressive_chain",
                "result_range": out_range,
                "counterfactual": spec["counterfactual"] if action=="counterfactual_forecast" else None
            }

    # threshold question
    if action == "threshold_eta":
        wc = tinfo["window_minutes"]
        return {
            "type": "threshold_eta",
            "machines": spec["machines"],
            "threshold": spec["threshold"],
            "eta_minutes_range": [wc["start_min"], wc["end_min"]]
        }

    return {
        "type": "unsupported",
        "raw_spec": spec
    }

