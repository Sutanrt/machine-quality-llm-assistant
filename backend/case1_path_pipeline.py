import pandas as pd
from pathlib import Path

# ===== internal modules =====
from Case1.data_loader import build_df_raw
from Case1.nlp_intent import interpret, extract_subintent
from Case1.machine_analyzer import analyze_trend_per_machine_v2, pretty_per_machine_v2
from Case1.plotter import serve_insight_with_charts
from Case1.model_runner import runner  # <-- ini triple-head DualTripleRunner yg sudah diload modelnya


# =========================================
# CORE PIPELINE / API FUNCTION
# =========================================

def run_query(
    query_text: str,
    df_raw: pd.DataFrame,
    plots_dir: str = "./plots",
    decimal_comma: bool = True,
    places: int = 6,
):
    """
    Langkah end-to-end dari natural language ke output analitik.

    Input:
      - query_text: str       -> "Cek tren mesin XP888A hari ini"
      - df_raw: DataFrame     -> hasil build_df_raw(), punya kolom machine_id | ts | value
      - plots_dir: str        -> folder tempat PNG disimpan
      - decimal_comma: bool   -> True = pakai koma untuk desimal (format lokal)
      - places: int           -> berapa digit desimal di pretty text

    Return dict siap pakai buat UI / logging:
    {
        "query": ...,
        "intent": {...},
        "human_report": "...",          # kalimat ringkasan buat operator/manajemen
        "llm_metadata": {...},          # angka terstruktur per mesin (buat LLM/dashboard)
        "plot_paths": {machine_id: [png1,png2,...]},
        "skipped_plots": False / True
    }
    """

    # 1. Prediksi intent high-level pakai model triple-head (scope/time/case)
    #    runner diimpor dari model_runner.py
    model_pred = runner.predict(query_text)
    # model_pred bentuknya:
    # {
    #   "scope": {"label": "ALL"/"NOT_ALL", ...},
    #   "time":  {"label": "DAY"/"HOUR"/... , ...},
    #   "case":  {"label": "SINGLE_PERIOD"/... , ...},
    #   "route_hints": [...]
    # }

    # 2. Post-process intent: gabung info waktu/nama mesin dari bahasa natural
    #    interpret() juga convert "NOT_ALL" -> "SUBSET" dsb
    intent = interpret(query_text, model_pred)

    # 3. Subintent detail apa yang user minta (trend/min/max/ringkasan/fluktuasi)
    subintent = extract_subintent(query_text)
    # contoh subintent:
    #   "trend", "minmax", "summary", "max_only", "min_only", ...

    # 4. Analisis statistik per mesin (tren, CV, outlier, periode fluktuasi, dsb)
    report = analyze_trend_per_machine_v2(
        df_raw,
        intent,
        ts_col="ts",
        machine_col="machine_id",
        value_col="value",
        agg="mean",
        z_th=3.0,
        rz_th=3.0,
        k_mad=4.5,
        cv_th=0.10,
    )

    # 5. Buat ringkasan teks human-readable untuk operator/manajemen
    human_text = pretty_per_machine_v2(
        report,
        decimal_comma=decimal_comma,
        places=places,
        show_outliers=3,
        show_windows=3,
    )

    # 6. Generate visualisasi (PNG line raw, line resampled, histogram)
    #    + paket metadata angka yang rapi untuk LLM/dashboard
    served = serve_insight_with_charts(
        df_raw,
        intent,
        report_fn=analyze_trend_per_machine_v2,
        subintent=subintent,
        out_dir=plots_dir,
        ts_col="ts",
        machine_col="machine_id",
        value_col="value",
    )

    # kasus gak ada data (misal range tanggal kosong)
    if served.get("empty"):
        return {
            "query": query_text,
            "intent": intent,
            "human_report": f"Tidak ada data. {served.get('reason', '')}",
            "llm_metadata": None,
            "plot_paths": {},
            "skipped_plots": True,
        }

    # 7. Bundle final supaya gampang dikonsumsi caller
    return {
        "query": query_text,
        "intent": intent,
        "human_report": human_text,
        "llm_metadata": served["meta"],        # dict numerik ringkas per mesin
        "plot_paths": served["plot_paths"],    # dict: {MESIN: ["./plots/MESIN_raw.png", ...]}
        "skipped_plots": served["skipped_plots"],
    }


# =========================================
# CLI / local debug mode
# =========================================
# Jalankan: python main_pipeline.py
# Ini akan:
#   - load data dari Excel -> df_raw
#   - simpan df_raw ke CSV lokal (arsip)
#   - jalankan beberapa contoh query
#   - print hasil ringkasan + lokasi plot PNG

