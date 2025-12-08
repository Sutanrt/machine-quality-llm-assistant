from typing import Dict, Any, List

# asumsi dua helper ini di-import dari rollout.py
# from .rollout import rollout_autoregressive, build_range_targets

def execute_spec(
    spec: Dict[str, Any],
    forecaster,
    context_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Jalankan spesifikasi forecast / threshold yang sudah dinormalisasi oleh interpret_query.

    Parameters
    ----------
    spec : dict
        NormalizedSpec dari interpret_query(), berisi:
        - action_type: str
        - machines: List[str]
        - time: {
            "time_complexity": "SINGLE_POINT" | "RANGE_REL" | "RANGE_FUTURE",
            "window_minutes": {"start_min": int, "end_min": int},
            ... (boleh ada multi_targets_minutes dll.)
          }
        - counterfactual: {...} atau None
        - threshold: {...} atau None (buat threshold_eta)

    forecaster : object
        Model / wrapper yang punya method predict(next_state) dsb.
        Biasanya HorizonForecasterClassic atau forecaster lain di pipeline kamu.

    context_state : dict
        State awal lag / konteks mesin pada timestamp anchor.
        Misalnya berisi last_known_values, rolling features, dsb.

    Returns
    -------
    dict
        Hasil siap-diserialkan (misal ke JSON API).
    """

    action = spec.get("action_type")
    tinfo  = spec.get("time", {})
    wc     = tinfo.get("window_minutes", {})
    start_m = wc.get("start_min")
    end_m   = wc.get("end_min")

    # ----- FORECAST-style actions -----
    if action in ["forecast_numeric", "counterfactual_forecast", "forecast_trend"]:
        tc = tinfo.get("time_complexity")

        # CASE A: user minta 1 titik waktu (misal "2 jam lagi berapa?")
        if tc == "SINGLE_POINT":
            # autoregressive sampai end_m menit ke depan
            rollout_res = rollout_autoregressive(
                forecaster=forecaster,
                context_state=context_state,
                minutes_ahead=end_m,
            )

            return {
                "action_type": action,
                "machines": spec.get("machines", []),
                "mode": "autoregressive_chain",
                "time_request": {
                    "target_request_min": end_m,
                    "time_complexity": tc,
                },
                "result_single": {
                    "final_pred": rollout_res["final_pred"],
                    "t_final": rollout_res["t_final"],
                    "trajectory": rollout_res["trajectory"],
                },
                "counterfactual": spec.get("counterfactual"),
            }

        # CASE B: user minta rentang ("30-120 menit ke depan", dsb.)
        # RANGE_REL / RANGE_FUTURE
        targets: List[int] = build_range_targets(start_m, end_m)

        timeline = []
        for tmin in targets:
            r = rollout_autoregressive(
                forecaster=forecaster,
                context_state=context_state,
                minutes_ahead=tmin,
            )
            timeline.append({
                "minute_ahead": tmin,
                "pred_value": r["final_pred"],
                "t_simulated": r["t_final"],
                "trajectory": r["trajectory"],
            })

        return {
            "action_type": action,
            "machines": spec.get("machines", []),
            "mode": "autoregressive_chain",
            "time_request": {
                "window_minutes": [start_m, end_m],
                "time_complexity": tc,
            },
            "timeline": timeline,
            "counterfactual": spec.get("counterfactual"),
        }

    # ----- THRESHOLD ETA -----
    if action == "threshold_eta":
        # Kapan ambang batas akan dilanggar, dalam rentang [start_m, end_m]
        return {
            "action_type": "threshold_eta",
            "machines": spec.get("machines", []),
            "threshold": spec.get("threshold"),
            "time_request": {
                "eta_minutes_range": [start_m, end_m],
                "time_complexity": tinfo.get("time_complexity"),
            },
        }

    # fallback kalau belum di-handle
    return {
        "action_type": "unsupported",
        "raw_spec": spec,
    }

# def execute_spec(spec, forecaster, context_dict):
#     action = spec["action_type"]
#     tinfo  = spec["time"]
#     wc     = tinfo["window_minutes"]
#     start_m = wc["start_min"]
#     end_m   = wc["end_min"]

#     # FORECAST_NUMERIC & COUNTERFACTUAL_FORECAST → numeric future
#     if action in ["forecast_numeric", "counterfactual_forecast", "forecast_trend"]:
#         tc = tinfo["time_complexity"]

#         if tc == "SINGLE_POINT":
#             # gunakan autoregressive rollout sampai end_m menit
#             rollout = rollout_autoregressive(forecaster, context_dict, end_m)
#             return {
#                 "type": action,
#                 "machines": spec["machines"],
#                 "mode": "autoregressive_chain",
#                 "target_request_min": end_m,
#                 "result_single": {
#                     "final_pred": rollout["final_pred"],
#                     "t_final": rollout["t_final"],
#                     "trajectory": rollout["trajectory"]
#                 },
#                 "counterfactual": spec["counterfactual"] if action=="counterfactual_forecast" else None
#             }

#         else:
#             # RANGE_REL / RANGE_FUTURE
#             # untuk range kita bisa:
#             # - generate grid titik waktu
#             # - untuk setiap titik waktu jalankan rollout_autoregressive sendiri sampai titik itu
#             # Ini mahal tapi jelas. (bisa di-cache)
#             targets = build_range_targets(start_m, end_m)
#             timeline = []
#             for tmin in targets:
#                 rollout = rollout_autoregressive(forecaster, context_dict, tmin)
#                 timeline.append({
#                     "minute_ahead": tmin,
#                     "pred_value": rollout["final_pred"],
#                     "t_simulated": rollout["t_final"],
#                     "trajectory": rollout["trajectory"]
#                 })

#             return {
#                 "type": action,
#                 "machines": spec["machines"],
#                 "mode": "autoregressive_chain",
#                 "window": [start_m, end_m],
#                 "timeline": timeline,
#                 "counterfactual": spec["counterfactual"] if action=="counterfactual_forecast" else None
#             }

#     # THRESHOLD_TIME tetap sama seperti sebelumnya
#     if action == "threshold_eta":
#         return {
#             "type": "threshold_eta",
#             "machines": spec["machines"],
#             "threshold": spec["threshold"],
#             "eta_minutes_range": [wc["start_min"], wc["end_min"]],
#         }

#     return {
#         "type": "unsupported",
#         "raw_spec": spec
#     }
