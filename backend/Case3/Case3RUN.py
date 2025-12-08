# main.py
from case3_pipeline.config import OUT_ROOT, FORECASTER_ROOT, DATA_CSV
from case3_pipeline.nlu_runner import ForecastMultiRunner
from case3_pipeline.forecaster import HorizonForecasterClassic
from case3_pipeline.timeseries_state import load_raw_timeseries
from case3_pipeline.orchestrator import run_full_query

def bootstrap():
    # 1. load model NLU (HF heads)
    runner = ForecastMultiRunner(root_dir=OUT_ROOT)

    # 2. load model klasik (linear+RF per horizon)
    forecaster = HorizonForecasterClassic(run_root=FORECASTER_ROOT)

    # 3. load data timeseries mentah
    df_raw = load_raw_timeseries(DATA_CSV)

    return runner, forecaster, df_raw

if __name__ == "__main__":
    runner, forecaster, df_raw = bootstrap()

    user_text = "kapan XP888A bakal turun di bawah 70 derajat? kalau iya jam berapa pertama kalinya?"
    result = run_full_query(
        user_text=user_text,
        runner=runner,
        df_raw=df_raw,
        forecaster=forecaster,
        n_lags=3,
    )

    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
