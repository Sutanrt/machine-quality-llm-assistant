# executor.py
from __future__ import annotations
from typing import Dict, Any, List

from .rollout import (
    rollout_autoregressive,
    rollout_incremental_horizons,
    build_range_targets,
)


def _forecast_point_autoreg(
    spec: Dict[str, Any],
    forecaster,
    context_state: Dict[str, float],
) -> Dict[str, Any]:
    """
    SINGLE_POINT case: "2 jam lagi berapa?"
    """
    target_min = spec["time"]["window_minutes"]["end_min"]
    rollout_res = rollout_autoregressive(forecaster, context_state, target_min)

    return {
        "target_request_min": target_min,
        "final_pred": rollout_res["final_pred"],
        "t_final": rollout_res["t_final"],
        "trajectory": rollout_res["trajectory"],
    }


def _forecast_range_autoreg(
    spec: Dict[str, Any],
    forecaster,
    context_state: Dict[str, float],
) -> Dict[str, Any]:
    """
    RANGE_REL / RANGE_FUTURE:
    "0 sampai 30 menit ke depan"
    Kita sampling beberapa titik waktu penting di window,
    jalankan rollout_autoregressive terpisah per titik.
    """
    w = spec["time"]["window_minutes"]
    start_m = w["start_min"]
    end_m   = w["end_min"]

    targets = build_range_targets(start_m, end_m)

    timeline_points: List[Dict[str, Any]] = []
    for tmin in targets:
        r = rollout_autoregressive(forecaster, context_state, tmin)
        timeline_points.append({
            "minute_ahead": tmin,
            "pred_value": r["final_pred"],
            "t_final": r["t_final"],
            "trajectory": r["trajectory"],
        })

    return {
        "window": [start_m, end_m],
        "timeline": timeline_points,
    }


def execute_spec(
    spec: Dict[str, Any],
    forecaster,
    context_state: Dict[str, float] | None,
) -> Dict[str, Any]:
    """
    spec:
        NormalizedSpec hasil interpret_query()
    forecaster:
        HorizonForecasterClassic (atau compatible)
    context_state:
        dict lag realtime mesin tertentu:
          {"y_lag_1":105.2,"y_lag_2":104.8,"y_lag_3":104.5}
        Bisa None kalau scope ALL tapi kita belum pilih mesin tertentu,
        tapi normalnya orchestrator bakal panggil per-mesin.
    return:
        dict hasil final siap konsumsi UI.
    """

    action = spec["action_type"]
    tinfo  = spec["time"]
    tc     = tinfo["time_complexity"]

    # Safety guard
    if context_state is None:
        return {
            "type": "no_context_state",
            "reason": "No machine context state available for forecasting.",
            "machines": spec.get("machines", []),
        }

    # MULTI-HORIZON query explicit: "12, 24, 48 menit"
    multi_list = tinfo.get("multi_targets_minutes")
    if multi_list and len(multi_list) > 1:
        reqs = sorted(multi_list)

        mh_res = rollout_incremental_horizons(
            forecaster,
            context_state,
            reqs,
        )

        return {
            "type": "forecast_multi_point",
            "machines": spec["machines"],
            "requested_minutes": reqs,
            "per_horizon": {
                str(t): {
                    "pred_value": mh_res[t]["pred_value"],
                    "t_effective": mh_res[t]["t_effective"],
                }
                for t in reqs
            },
        }

    # Forecast-style intent
    if action in ["forecast_numeric", "forecast_trend", "counterfactual_forecast"]:
        if tc == "SINGLE_POINT":
            out_single = _forecast_point_autoreg(spec, forecaster, context_state)
            return {
                "type": action,
                "machines": spec["machines"],
                "mode": "autoregressive_chain",
                "result_single": out_single,
                "counterfactual": (
                    spec["counterfactual"] if action == "counterfactual_forecast" else None
                ),
            }

        # RANGE_REL / RANGE_FUTURE dll
        out_range = _forecast_range_autoreg(spec, forecaster, context_state)
        return {
            "type": action,
            "machines": spec["machines"],
            "mode": "autoregressive_chain",
            "result_range": out_range,
            "counterfactual": (
                spec["counterfactual"] if action == "counterfactual_forecast" else None
            ),
        }

    # Threshold / ETA intent
    if action == "threshold_eta":
        wc = tinfo["window_minutes"]
        return {
            "type": "threshold_eta",
            "machines": spec["machines"],
            "threshold": spec["threshold"],
            "eta_minutes_range": [wc["start_min"], wc["end_min"]],
        }

    # fallback
    return {
        "type": "unsupported",
        "raw_spec": spec,
    }
