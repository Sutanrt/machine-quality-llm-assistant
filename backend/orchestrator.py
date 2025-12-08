# orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import pandas as pd

from .nlu_runner import ForecastMultiRunner
from .interpret_query import interpret_query
from .executor import execute_spec
from .timeseries_state import (
    build_initial_lag_state,
    get_all_machine_ids,
)


def _resolve_anchor_time(anchor_time_type: str) -> Optional[str]:
    """
    Terjemahkan tipe anchor waktu ke timestamp anchor yang mau dipakai
    untuk build_initial_lag_state.
    Saat ini:
    - "NOW_START" → None  (artinya pakai newest row)
    - lainnya     → None  (belum parse natural language absolute time)
    """
    if anchor_time_type == "NOW_START":
        return None
    return None  # TODO: parse absolute time phrases later


def _collect_target_machines(spec: Dict[str, Any], df_raw: pd.DataFrame) -> List[str]:
    """
    Nentuin mesin mana aja yang harus diproses berdasarkan scope.

    - scope == "SINGLE"
        ambil mesin pertama yg bukan "UNKNOWN", kalau gak ada fallback pertama
    - scope == "ALL"
        semua mesin unik di df_raw
    - scope == "MULTI"
        semua mesin yg user sebut (tanpa "UNKNOWN"); kalau kosong fallback all
    - fallback
        ambil mesin pertama dari df_raw (supaya gak crash)
    """
    scope = spec.get("scope")
    mentioned = spec.get("machines", [])

    if scope == "ALL":
        return get_all_machine_ids(df_raw)

    if scope == "SINGLE":
        for m in mentioned:
            if m != "UNKNOWN":
                return [m]
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
        return get_all_machine_ids(df_raw)

    # fallback generic
    all_ids = get_all_machine_ids(df_raw)
    return [all_ids[0]] if all_ids else []


def run_full_query(
    user_text: str,
    runner: ForecastMultiRunner,
    df_raw: pd.DataFrame,
    forecaster,
    n_lags: int = 3,
) -> Dict[str, Any]:
    """
    Full pipeline buat satu pertanyaan user:
    1. runner.predict() → info intent/scope/time/horizon/etc.
    2. interpret_query() → NormalizedSpec siap dieksekusi
    3. cari anchor time (NOW_START → None, dll.)
    4. tentuin mesin target
    5. untuk tiap mesin:
        - build_initial_lag_state()
        - execute_spec()
    6. kumpulin hasil jadi array of results

    Output schema:
    {
      "runner_out": {...},    # raw NLU
      "spec": {...},          # NormalizedSpec
      "results": [
        {
          "machine_id": "XP888A",
          "exec_out": {
             ... hasil execute_spec(),
             "anchor_used_ts": "...",
             "machine_context": "XP888A"
          }
        },
        ...
      ]
    }
    """

    # 1. klasifikasi + slot filling
    runner_out = runner.predict(user_text)

    # 2. interpretasi ke spec normalized
    spec = interpret_query(user_text, runner_out)

    # 3. anchor time dari spec
    anchor_time_type = spec["time"]["anchor_time_type"]
    anchor_time = _resolve_anchor_time(anchor_time_type)

    # 4. daftar mesin target
    target_machines = _collect_target_machines(spec, df_raw)

    # 5. loop setiap mesin dan eksekusi forecast
    results: List[Dict[str, Any]] = []

    for mid in target_machines:
        try:
            init_state, anchor_used_ts = build_initial_lag_state(
                df_raw=df_raw,
                machine_id=mid,
                anchor_time=anchor_time,
                n_lags=n_lags,
            )

            # untuk mesin spesifik ini, kita mau kasih nama mesin bener ke executor.
            # executor gak butuh scope "ALL", jadi aman kalau kita clone spec:
            spec_m = json.loads(json.dumps(spec))
            spec_m["machines"] = [mid]

            # opsional: normalize scope jadi "SINGLE" supaya downstream jelas
            spec_m["scope"] = "SINGLE"

            exec_out = execute_spec(
                spec_m,
                forecaster=forecaster,
                context_state=init_state,
            )

            exec_out["anchor_used_ts"] = str(anchor_used_ts)
            exec_out["machine_context"] = mid

            results.append({
                "machine_id": mid,
                "exec_out": exec_out,
            })

        except Exception as e:
            # Mesin ini gagal (data kosong, dsb). Kita tangkap dan lanjut.
            results.append({
                "machine_id": mid,
                "error": str(e),
            })

    return {
        "runner_out": runner_out,
        "spec": spec,
        "results": results,
    }
