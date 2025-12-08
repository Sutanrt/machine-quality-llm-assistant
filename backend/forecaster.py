# forecaster.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import joblib

# mapping horizon head -> menit ke depan
HORIZON_MINUTES = {
    "y_step_3m": 3,
    "y_step_6m": 6,
    "y_step_12m": 12,
    "y_step_24m": 24,
    "y_step_48m": 48,
}

# kebalikan buat lookup cepat: 12  -> "y_step_12m"
HORIZON_BY_MIN = {v: k for k, v in HORIZON_MINUTES.items()}


class HorizonForecasterClassic:
    """
    Loader + predictor untuk model klasik kombinasi:
    linear model + random forest residual per horizon.
    Folder run_root diasumsikan punya subfolder per horizon:
      run_root/
        y_step_12m/
           y_step_12m_linear.joblib
           y_step_12m_rf_residual.joblib
           y_step_12m_config.json  (berisi 'use_lags': ["y_lag_1", ...])
    """

    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _load_horizon(self, horizon_key: str) -> Dict[str, Any]:
        """
        Baca model linear, RF residual, dan daftar fitur lag untuk horizon_key.
        Cache supaya gak reload setiap prediksi.
        """
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

    def predict_single_horizon(self, horizon_key: str, state_lags: Dict[str, float]) -> float:
        """
        horizon_key: e.g. 'y_step_12m'
        state_lags: dict kondisi sekarang:
            {
              "y_lag_1": <float latest>,
              "y_lag_2": <float prev>,
              "y_lag_3": <float prevprev>,
              ...
            }
        return: float prediksi gabungan linear + RF residual
        """
        bundle = self._load_horizon(horizon_key)
        use_lags = bundle["use_lags"]

        X_now = np.array([[ state_lags[feat] for feat in use_lags ]], dtype=float)

        y_lin = bundle["lin"].predict(X_now)[0]
        y_res = bundle["rf"].predict(X_now)[0]
        y_combo = float(y_lin + y_res)
        return y_combo
