# orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from .nlu_runner import ForecastMultiRunner
from .interpret_query import interpret_query
from .executor import execute_spec
from .timeseries_state import build_initial_lag_state


def get_all_machine_ids(df_raw: pd.DataFrame) -> List[str]:
    """
    Ambil daftar semua machine_id unik dari dataframe sensor mentah.
    Diasumsikan kolomnya bernama 'machine_id'.
    """
    return sorted(df_raw["machine_id"].unique().tolist())


def _resolve_anchor_time(anchor_time_type: str) -> Optional[str]:
    """
    Translasi tipe anchor waktu → timestamp anchor yang mau dipakai
    buat ngambil lag state.

    Sekarang:
    - "NOW_START" -> None  (artinya pakai paling recent row per mesin)
    - lainnya     -> None  (TODO: nanti parse natural language seperti
                            "kemarin jam 10")
    """
    if anchor_time_type == "NOW_START":
        return None
    # TODO: parse absolute time text kalau kamu nanti simpan di spec
    return None


def _collect_target_machines(
    spec: Dict[str, Any],
    df_raw: pd.DataFrame,
) -> List[str]:
    """
    Tentukan mesin mana yang akan kita proses berdasarkan scope di spec.

    Rules:
    - scope == "ALL"
        -> semua machine_id unik dari df_raw
    - scope == "SINGLE"
        -> mesin pertama yang disebut di spec["machines"]
           (kecuali "UNKNOWN", kita skip itu dan cari yang valid)
    - scope == "MULTI"
        -> semua mesin yang disebut user di spec["machines"],
           buang "UNKNOWN", dedup.
           Kalau kosong -> fallback ke semua mesin.
    - default / gak jelas
        -> fallback: ambil machine pertama di df_raw biar nggak error.
    """
    scope = spec.get("scope")
    mentioned = spec.get("machines", [])

    if scope == "ALL":
        return get_all_machine_ids(df_raw)

    if scope == "SINGLE":
        # pilih mesin pertama yang bukan 'UNKNOWN'
        for m in mentioned:
            if m != "UNKNOWN":
                return [m]
        # kalau gak ada yang jelas, fallback aja ke elemen pertama
        return [mentioned[0]] if mentioned else []

    if scope == "MULTI":
        cleaned = []
        for m in mentioned:
            if m == "UNKNOWN":
                continue
            if m not in cleaned:
                cleaned.append(m)
        if cleaned:
            return cleaned
        # fallback
        return get_all_machine_ids(df_raw)

    # fallback terakhir: ambil mesin pertama di data
    all_ids = get_all_machine_ids(df_raw)
    return [all_ids[0]] if all_ids else []


def _forecast_for_machine(
    spec: Dict[str, Any],
    machine_id: str,
    df_raw: pd.DataFrame,
    forecaster,
    n_lags: int,
    anchor_time: Optional[str],
) -> Dict[str, Any]:
    """
    - Bangun initial lag state untuk machine_id tertentu
    - Panggil execute_spec()
    - Tambahkan metadata (anchor_used_ts & machine_id)
    """
    init_state, anchor_used_ts = build_initial_lag_state(
        df_raw=df_raw,
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


def run_full_query(
    user_text: str,
    runner: ForecastMultiRunner,   # NLU classifier (intent/scope/time/etc.)
    df_raw: pd.DataFrame,          # dataframe time-series mentah
    forecaster,                    # HorizonForecasterClassic / GRU / dll.
    n_lags: int = 3,
) -> Dict[str, Any]:
    """
    Jalankan full pipeline end-to-end untuk satu pertanyaan user.

    Steps:
    1. NLU -> runner.predict(user_text)
    2. interpret_query() -> NormalizedSpec
    3. Tentukan anchor time (misal "NOW_START")
    4. Tentukan target machines (SINGLE / MULTI / ALL)
    5. Untuk tiap mesin:
        - build_initial_lag_state()
        - execute_spec()
    6. Gabungkan hasil per mesin

    Return shape:
    {
      "runner_out": {...},     # raw output dari ForecastMultiRunner
      "spec": {...},           # NormalizedSpec final
      "results": [
        {
          "machine_id": "XP888A",
          "exec_out": {...}   # hasil execute_spec() + anchor_used_ts dsb.
        },
        ...
      ]
    }
    """

    # 1. Klasifikasi intent/scope/time/horizon dsb pakai semua head NLU
    runner_out = runner.predict(user_text)

    # 2. Normalisasi → spesifikasi eksekusi yang stabil
    spec = interpret_query(user_text, runner_out)

    # 3. Cari timestamp anchor dari spec["time"]["anchor_time_type"]
    anchor_time_type = spec["time"]["anchor_time_type"]
    anchor_time = _resolve_anchor_time(anchor_time_type)

    # 4. Tentukan mesin mana saja perlu dieksekusi
    target_machines = _collect_target_machines(spec, df_raw)

    # 5. Loop setiap mesin, run execute_spec
    results = []
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
            results.append({
                "machine_id": machine_id,
                "exec_out": exec_out,
            })
        except Exception as e:
            # jangan biarkan satu mesin nge-crash semuanya
            results.append({
                "machine_id": machine_id,
                "error": str(e),
            })

    return {
        "runner_out": runner_out,
        "spec": spec,
        "results": results,
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
