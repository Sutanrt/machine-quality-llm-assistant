# orchestrator.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

from .nlu_runner import ForecastMultiRunner
from .interpret_query import interpret_query
from .executor import execute_spec
# fungsi ini kamu punya di modul timeseries_state.py (kita rencanakan)
# - build_initial_lag_state(df_raw, machine_id, anchor_time, n_lags)
# - get_all_machine_ids(df_raw)
from .timeseries_state import build_initial_lag_state, get_all_machine_ids


def _resolve_anchor_time(anchor_time_type: str) -> Optional[str]:
    """
    Placeholder parser waktu natural language → timestamp anchor.
    Sekarang:
    - "NOW_START"      -> None  (artinya pakai row/latest timestamp di data)
    - selain itu       -> None (belum diimplementasi parsing absolut)
    Nanti bisa di-upgrade buat parse "kemarin jam 10".
    """
    if anchor_time_type == "NOW_START":
        return None
    # TODO: parse absolute time from user text if provided
    return None


def _forecast_for_machine(
    spec: Dict[str, Any],
    machine_id: str,
    df_raw: pd.DataFrame,
    forecaster,
    n_lags: int = 3,
    anchor_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate initial lag state for a machine, run execute_spec(),
    and decorate result with metadata.
    """
    init_state, anchor_used_ts = build_initial_lag_state(
        df_raw,
        machine_id=machine_id,
        anchor_time=anchor_time,
        n_lags=n_lags,
    )

    exec_out = execute_spec(
        spec,
        forecaster=forecaster,
        context_state=init_state,
    )

    exec_out["anchor_used_ts"] = (
        str(anchor_used_ts) if anchor_used_ts is not None else None
    )
    exec_out["machine_context"] = machine_id

    return exec_out


def _collect_target_machines(
    spec: Dict[str, Any],
    df_raw: pd.DataFrame,
) -> List[str]:
    """
    Tentukan mesin mana saja yang akan diproses,
    berdasarkan scope & hasil interpretasi mesin di spec.

    Rules:
    - scope == "ALL"   -> semua machine_id unik dari df_raw
    - scope == "SINGLE"-> ambil mesin pertama dari spec["machines"]
    - scope == "MULTI" -> pakai list spec["machines"] (kecuali "UNKNOWN")
    - fallback         -> pakai mesin pertama yang valid
    """

    scope = spec.get("scope")
    mentioned = spec.get("machines", [])

    if scope == "ALL":
        return get_all_machine_ids(df_raw)

    if scope == "SINGLE":
        # ambil mesin pertama yg bukan "UNKNOWN"
        for m in mentioned:
            if m != "UNKNOWN":
                return [m]
        return [mentioned[0]] if mentioned else []

    if scope == "MULTI":
        # keep order, drop UNKNOWN, unique
        cleaned = []
        for m in mentioned:
            if m == "UNKNOWN":
                continue
            if m not in cleaned:
                cleaned.append(m)
        # kalau setelah dibersihin kosong -> fallback semua?
        return cleaned if cleaned else get_all_machine_ids(df_raw)

    # default / tidak dikenal
    # fallback: mesin pertama di data (stabil untuk debug)
    all_ids = get_all_machine_ids(df_raw)
    if all_ids:
        return [all_ids[0]]
    return []


def run_full_query(
    user_text: str,
    runner: ForecastMultiRunner,
    df_raw: pd.DataFrame,
    forecaster,
    n_lags: int = 3,
) -> Dict[str, Any]:
    """
    Pipeline end-to-end untuk satu pertanyaan user:
    1. NLU (intent/scope/time/etc.) via ForecastMultiRunner
    2. interpret_query() → NormalizedSpec
    3. Tentukan anchor time & target machines
    4. Buat lag state, jalankan execute_spec() per mesin
    5. Balikkan paket lengkap

    Parameters
    ----------
    user_text : str
        Pertanyaan user dalam bahasa natural.
    runner : ForecastMultiRunner
        NLU head loader (intent/scope/time/etc.).
    df_raw : pd.DataFrame
        Time series mentah (punya kolom machine_id, timestamp, dsb.)
        Digunakan untuk bikin initial lag state.
    forecaster : object
        HorizonForecasterClassic / GRU / model prediksi nilai masa depan.
        Harus punya method predict_single_horizon(hk, state_dict).
    n_lags : int
        Berapa panjang lag state yang mau diambil dari df_raw.

    Returns
    -------
    dict
        {
          "runner_out": {...},  # keluaran ForecastMultiRunner.predict()
          "spec": {...},        # NormalizedSpec dari interpret_query()
          "results": [          # list per machine
            {
              "machine_id": "...",
              "exec_out": {...} # output execute_spec() + metadata anchor
            },
            ...
          ]
        }
    """

    # 1. classify intent/scope/time/etc.
    runner_out = runner.predict(user_text)

    # 2. build NormalizedSpec dari interpret_query
    spec = interpret_query(user_text, runner_out)

    # 3. tentukan anchor_time (misal NOW_START -> None)
    anchor_time_type = spec["time"]["anchor_time_type"]
    anchor_time = _resolve_anchor_time(anchor_time_type)

    # 4. figure out target machines to forecast
    target_machines = _collect_target_machines(spec, df_raw)

    results_per_machine: List[Dict[str, Any]] = []

    for machine_id in target_machines:
        try:
            exec_out = _forecast_for_machine(
                spec=spec,
                machine_id=machine_id,
                df_raw=df_raw,
                forecaster=forecaster,
                n_lags=n_lags,
                anchor_time=anchor_time,
            )

            results_per_machine.append({
                "machine_id": machine_id,
                "exec_out": exec_out,
            })

        except Exception as e:
            # kita tangkap error per mesin supaya mesin lain tetap jalan
            results_per_machine.append({
                "machine_id": machine_id,
                "error": str(e),
            })

    return {
        "runner_out": runner_out,
        "spec": spec,
        "results": results_per_machine,
    }

# def get_all_machine_ids(df_raw: pd.DataFrame):
#     return sorted(df_raw["machine_id"].unique().tolist())

# def run_full_query(user_text,
#                    runner,          # ForecastMultiRunner (NLP classifier)
#                    df_raw,          # dataframe hasil load_raw_timeseries(...)
#                    forecaster):     # HorizonForecaster / GRU forecaster

#     # 1. classify + slot fill
#     runner_out = runner.predict(user_text)
#     spec = interpret_query(user_text, runner_out)

#     # 2. tentukan mesin target utk state awal
#     #    - scope SINGLE → pakai machines[0]
#     #    - scope ALL → ??? (kita bisa pilih avg atau per-mesin loop)
#     if spec["scope"] == "SINGLE":
#         target_machine = spec["machines"][0]
#     else:
#         # sementara untuk ALL kita pilih salah satu mesin dulu / atau fail.
#         # Nanti kita loop ALL satu per satu.
#         target_machine = spec["machines"][0] if spec["machines"] else None

#     # 3. tentukan anchor time
#     #   - default: "NOW_START" → pakai timestamp terbaru (None in build fn)
#     #   - kalau spec['time']['anchor_time_type'] == 'ABSOLUTE_START'
#     #     dan ada spec['time']['anchor_time_text'] (misal "kemarin jam 10"),
#     #     kamu nanti parse itu -> datetime
#     if spec["time"]["anchor_time_type"] == "NOW_START":
#         anchor_time = None
#     else:
#         # TODO: parse natural language time "kemarin jam 10"
#         # sementara fallback None juga
#         anchor_time = None

#     # 4. build initial lag state dari CSV
#     if target_machine is not None:
#         init_state, anchor_used_ts = build_initial_lag_state(
#             df_raw,
#             machine_id=target_machine,
#             anchor_time=anchor_time,
#             n_lags=3
#         )
#     else:
#         # kasus ALL → kita mungkin belum define, taruh None dulu
#         init_state, anchor_used_ts = None, None

#     # 5. panggil executor dengan init_state tersebut
#     exec_out = execute_spec(
#         spec,
#         forecaster=forecaster,
#         context_state=init_state
#     )

#     # 6. enrich exec_out dengan metadata waktu supaya jelas ke user
#     exec_out["anchor_used_ts"] = str(anchor_used_ts) if anchor_used_ts is not None else None
#     exec_out["machine_context"] = target_machine

#     return {
#         "runner_out": runner_out,
#         "spec": spec,
#         "exec_out": exec_out,
#     }
