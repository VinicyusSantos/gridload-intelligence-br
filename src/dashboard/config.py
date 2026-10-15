"""Configuracoes centralizadas da aplicacao de portfolio."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BASE_PATH = DATA_DIR / "processed/base.parquet"
FEATURED_BASE_PATH = DATA_DIR / "processed/base_final.parquet"
FORECASTS_PATH = DATA_DIR / "predictions/latest_forecasts.parquet"
FORECASTS_JSON_PATH = DATA_DIR / "predictions/latest_forecasts.json"
GLOBAL_SHAP_PATH = DATA_DIR / "explainability/global_shap.parquet"
MODEL_COMPARISON_PATH = DATA_DIR / "evaluation/model_comparison.parquet"
TEST_PREDICTIONS_PATH = DATA_DIR / "evaluation/test_predictions.parquet"
STYLE_PATH = PROJECT_ROOT / "assets/styles.css"

HORIZONS = ("1h", "6h", "24h")
RECENT_HISTORY_HOURS = 48
GLOBAL_SHAP_SAMPLE_SIZE = 1_000

HORIZON_COLORS = {
    "1h": "#2563EB",
    "6h": "#0F766E",
    "24h": "#7C3AED",
}

# Resultados finais do conjunto de teste do modelo tunado, consolidados em
# notebooks/07_explainability.ipynb. O sinal do vies e observado - previsto.
TEST_METRICS = {
    "1h": {
        "mae_mw": 149.63,
        "rmse_mw": 203.60,
        "mape_pct": 1.14,
        "r2": 0.976,
        "bias_mw": 2.69,
        "max_abs_error_mw": 1295.48,
        "feature_count": 22,
    },
    "6h": {
        "mae_mw": 399.32,
        "rmse_mw": 516.20,
        "mape_pct": 3.06,
        "r2": 0.849,
        "bias_mw": 1.48,
        "max_abs_error_mw": 2686.67,
        "feature_count": 29,
    },
    "24h": {
        "mae_mw": 370.85,
        "rmse_mw": 526.98,
        "mape_pct": 2.86,
        "r2": 0.843,
        "bias_mw": 35.46,
        "max_abs_error_mw": 3703.68,
        "feature_count": 15,
    },
}

DISCLAIMER = (
    "Projeto de Ciência de Dados desenvolvido para previsão de carga elétrica "
    "do subsistema Nordeste. As previsões possuem finalidade analítica e não "
    "representam recomendações de operação do Sistema Interligado Nacional."
)
