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

# Resultados do conjunto de teste fornecidos na especificacao da interface.
TEST_METRICS = {
    "1h": {"mae_mw": 149.63, "rmse_mw": 203.60, "mape_pct": 1.14},
    "6h": {"mae_mw": 399.32, "rmse_mw": 516.20, "mape_pct": 3.06},
    "24h": {"mae_mw": 303.44, "rmse_mw": 418.48, "mape_pct": 2.33},
}

DISCLAIMER = (
    "Projeto de Ciência de Dados desenvolvido para previsão de carga elétrica "
    "do subsistema Nordeste. As previsões possuem finalidade analítica e não "
    "representam recomendações de operação do Sistema Interligado Nacional."
)
