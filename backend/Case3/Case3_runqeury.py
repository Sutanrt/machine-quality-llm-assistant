import pandas as pd
from datetime import datetime
import json
from pathlib import Path

# BASE_DIR = folder backend/
BASE_DIR = Path(__file__).resolve().parent
def load_raw_timeseries(csv_path=BASE_DIR/"df_raw2.csv"):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["machine_id","ts"]).reset_index(drop=True)
    return df

def get_all_machine_ids(df_raw: pd.DataFrame):
    return sorted(df_raw["machine_id"].unique().tolist())

def build_initial_lag_state(df_raw: pd.DataFrame,
                            machine_id: str,
                            anchor_time: datetime | None = None,
                            n_lags: int = 3):
    df_m = df_raw[df_raw["machine_id"] == machine_id].copy()
    if df_m.empty:
        raise RuntimeError(f"no data for machine {machine_id}")

    if anchor_time is not None:
        df_m = df_m[df_m["ts"] <= anchor_time]

    df_m = df_m.sort_values("ts", ascending=False).reset_index(drop=True)

    if len(df_m) < 1:
        raise RuntimeError(f"machine {machine_id}: no rows before anchor_time")
    # fallback: kalau kurang dari n_lags, duplikasi nilai terakhir
    vals = df_m["value"].astype(float).tolist()
    while len(vals) < n_lags:
        vals.append(vals[-1])
    vals = vals[:n_lags]

    lag_state = {f"y_lag_{i+1}": float(v) for i,v in enumerate(vals)}
    anchor_used_ts = df_m.loc[0,"ts"]
    return lag_state, anchor_used_ts


def run_full_query(user_text, runner, df_raw, forecaster):
    """
    High level orchestrator:
    1. NLU (runner.predict)
    2. Interpret -> spec
    3. Jika SINGLE → prediksi satu mesin
       Jika ALL    → prediksi semua mesin yang ada di df_raw
    """

    # --- 1. classify + slot extract dari teks user
    runner_out = runner.predict(user_text)
    spec = interpret_query(user_text, runner_out)

    # --- 2. anchor time parsing
    # sekarang masih simple:
    # NOW_START -> pakai latest row
    # (kalau nanti ada ABSOLUTE_START kita parse natural datetime di sini)
    if spec["time"]["anchor_time_type"] == "NOW_START":
        anchor_time = None
    else:
        anchor_time = None  # TODO nanti parse spec["time"]["anchor_time_text"]

    # --- CASE A: scope == "SINGLE"
    if spec["scope"] == "SINGLE":
        target_machine = spec["machines"][0]

        init_state, anchor_used_ts = build_initial_lag_state(
            df_raw,
            machine_id=target_machine,
            anchor_time=anchor_time,
            n_lags=3,
        )

        exec_out = execute_spec(
            spec,
            forecaster=forecaster,
            context_state=init_state
        )

        exec_out["anchor_used_ts"]   = str(anchor_used_ts)
        exec_out["machine_context"]  = target_machine
        exec_out["scope_resolved"]   = "SINGLE"

        return {
            "runner_out": runner_out,
            "spec": spec,
            "exec_out": exec_out,
        }

    # --- CASE B: scope == "ALL"
    # Cara kita: loop setiap mesin real di df_raw, jalankan pipeline yang sama,
    # kumpulkan hasil dalam satu dict.
    elif spec["scope"] == "ALL":
        all_ids = get_all_machine_ids(df_raw)

        per_machine_result = {}
        for mid in all_ids:
            try:
                # 1. ambil lag actual utk mesin ini
                init_state, anchor_used_ts = build_initial_lag_state(
                    df_raw,
                    machine_id=mid,
                    anchor_time=anchor_time,
                    n_lags=3,
                )

                # 2. buat salinan spec khusus mesin ini
                #    kenapa? karena spec["machines"] sekarang ['ALL']
                #    tapi executor butuh nama mesin beneran untuk context kalau SINGLE-like
                #    kita bikin shallow copy aman
                spec_m = json.loads(json.dumps(spec))
                spec_m["machines"] = [mid]

                # NOTE:
                # kalau executor kamu butuh scope "ALL" buat range, biarin.
                # Tapi kalau executor kamu crash kalau scope != "SINGLE", 
                # kamu bisa force:
                # spec_m["scope"] = "SINGLE"
                # supaya code downstream treat sama seperti 1 mesin.
                #
                # Ini fleksibel tergantung versi execute_spec kamu sekarang.
                #
                # Rekomendasi awal: paksa jadi SINGLE.
                spec_m["scope"] = "SINGLE"

                # 3. run eksekusi forecast utk mesin ini
                machine_exec = execute_spec(
                    spec_m,
                    forecaster=forecaster,
                    context_state=init_state
                )

                # 4. lampirkan metadata anchor ts
                machine_exec["anchor_used_ts"]  = str(anchor_used_ts)
                machine_exec["machine_context"] = mid

                per_machine_result[mid] = machine_exec

            except Exception as e:
                # kalau satu mesin gagal (misal datanya terlalu sedikit),
                # kita tangkap supaya mesin lain tetap jalan
                per_machine_result[mid] = {
                    "error": str(e),
                    "machine_context": mid
                }

        final_exec_out = {
            "type":        "batch_forecast",
            "scope_resolved": "ALL",
            "machines":    all_ids,
            "per_machine": per_machine_result
        }

        return {
            "runner_out": runner_out,
            "spec": spec,
            "exec_out": final_exec_out,
        }

    # --- CASE C: MULTI (misal user sebut dua mesin spesifik)
    elif spec["scope"] == "MULTI":
        # spec["machines"] = ["PQ667","XP888A", ...]
        per_machine_result = {}
        for mid in spec["machines"]:
            try:
                init_state, anchor_used_ts = build_initial_lag_state(
                    df_raw,
                    machine_id=mid,
                    anchor_time=anchor_time,
                    n_lags=3,
                )

                spec_m = json.loads(json.dumps(spec))
                spec_m["machines"] = [mid]
                spec_m["scope"] = "SINGLE"

                machine_exec = execute_spec(
                    spec_m,
                    forecaster=forecaster,
                    context_state=init_state
                )

                machine_exec["anchor_used_ts"]  = str(anchor_used_ts)
                machine_exec["machine_context"] = mid

                per_machine_result[mid] = machine_exec

            except Exception as e:
                per_machine_result[mid] = {
                    "error": str(e),
                    "machine_context": mid
                }

        final_exec_out = {
            "type":        "batch_forecast",
            "scope_resolved": "MULTI",
            "machines":    spec["machines"],
            "per_machine": per_machine_result
        }

        return {
            "runner_out": runner_out,
            "spec": spec,
            "exec_out": final_exec_out,
        }

    else:
        # fallback sanity
        return {
            "runner_out": runner_out,
            "spec": spec,
            "exec_out": {
                "error": f"unhandled scope {spec['scope']}"
            }
        }
