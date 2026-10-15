"""Resultados historicos da modelagem exibidos pelo dashboard.

Os valores abaixo foram consolidados dos outputs preservados em
``notebooks/05_baseline.ipynb``, ``06_forecasting.ipynb`` e
``07_explainability.ipynb``. Eles descrevem o experimento temporal congelado;
nao sao metricas recalculadas para os modelos operacionais retreinados.
"""

from __future__ import annotations

import pandas as pd

from src.dashboard.config import HORIZONS, TEST_METRICS
from src.models.train_models import load_model_specs


EXPERIMENT_PERIOD = {
    "rows": 40_704,
    "start": "2022-01-08 00:00:00",
    "end": "2026-08-30 23:00:00",
    "train_rows": 28_492,
    "validation_rows": 6_106,
    "test_rows": 6_106,
    "test_start": "2025-12-19 14:00:00",
    "test_end": "2026-08-30 23:00:00",
}


BASELINE_RESULTS = (
    ("1h", "Persistencia", 294.02, 373.36, 2.35),
    ("1h", "Mesmo horario ontem", 595.95, 910.10, 4.93),
    ("1h", "Semana anterior", 465.35, 651.57, 3.75),
    ("6h", "Persistencia", 1123.66, 1358.56, 9.11),
    ("6h", "Mesmo horario ontem", 595.90, 910.05, 4.93),
    ("6h", "Semana anterior", 465.32, 651.53, 3.75),
    ("24h", "Persistencia", 596.66, 911.67, 4.93),
    ("24h", "Mesmo horario ontem", 596.66, 911.67, 4.93),
    ("24h", "Semana anterior", 465.32, 651.51, 3.75),
)


# Recalculo das mesmas regras somente nas 6.106 linhas do teste temporal. Esses
# valores permitem uma comparacao justa com as metricas finais do modelo; o
# notebook 05 originalmente reportou os baselines sobre as 40.704 linhas.
TEST_BASELINE_RESULTS = (
    ("1h", "Persistencia", 323.48, 411.66, 2.43),
    ("1h", "Mesmo horario ontem", 656.08, 968.77, 5.15),
    ("1h", "Semana anterior", 564.30, 775.97, 4.31),
    ("6h", "Persistencia", 1326.18, 1601.37, 9.99),
    ("6h", "Mesmo horario ontem", 656.00, 968.72, 5.15),
    ("6h", "Semana anterior", 563.91, 775.70, 4.31),
    ("24h", "Persistencia", 661.13, 979.30, 5.18),
    ("24h", "Mesmo horario ontem", 661.13, 979.30, 5.18),
    ("24h", "Semana anterior", 564.51, 776.16, 4.31),
)


VALIDATION_RESULTS = (
    ("1h", "Regressao linear", "Completo", 204.85, 268.27, 1.57),
    ("1h", "Random Forest", "Lags", 176.68, 242.77, 1.36),
    ("1h", "XGBoost base", "Janelas", 147.28, 205.90, 1.13),
    ("6h", "Regressao linear", "Completo", 451.03, 586.75, 3.56),
    ("6h", "Random Forest", "Meteorologia", 376.40, 492.70, 2.91),
    ("6h", "XGBoost base", "Meteorologia", 369.07, 477.00, 2.86),
    ("24h", "Regressao linear", "Completo", 490.28, 664.06, 3.88),
    ("24h", "Random Forest", "Lags", 308.13, 434.16, 2.37),
    ("24h", "XGBoost base", "Lags", 303.44, 418.48, 2.33),
)


TUNING_RESULTS = (
    ("1h", "XGBoost selecionado", "Validacao", 147.28, 205.90, 1.13),
    ("1h", "Early stopping", "Validacao", 134.73, 188.59, 1.03),
    ("1h", "XGBoost tunado", "Validacao", 139.65, 193.69, 1.08),
    ("1h", "Teste final", "Teste", 149.63, 203.60, 1.14),
    ("6h", "XGBoost selecionado", "Validacao", 369.07, 477.00, 2.86),
    ("6h", "Early stopping", "Validacao", 344.32, 445.36, 2.66),
    ("6h", "XGBoost tunado", "Validacao", 345.69, 444.72, 2.67),
    ("6h", "Teste final", "Teste", 399.32, 516.20, 3.06),
    ("24h", "XGBoost selecionado", "Validacao", 303.44, 418.48, 2.33),
    ("24h", "Early stopping", "Validacao", 301.11, 418.92, 2.31),
    ("24h", "XGBoost tunado", "Validacao", 305.79, 419.24, 2.35),
    ("24h", "Teste final", "Teste", 370.85, 526.98, 2.86),
)


WORST_CASES = (
    ("1h", "2026-01-30 16:00:00", 14603.15, 13307.67, 1295.48, "Subestimacao"),
    ("6h", "2025-12-31 09:00:00", 12303.17, 14989.84, 2686.67, "Superestimacao"),
    ("24h", "2026-04-02 13:00:00", 8889.52, 12593.20, 3703.68, "Superestimacao"),
)


def test_metrics_frame() -> pd.DataFrame:
    """Retorna as metricas finais do teste na ordem dos horizontes."""
    return pd.DataFrame(
        [{"horizon": horizon, **TEST_METRICS[horizon]} for horizon in HORIZONS]
    )


def baseline_metrics_frame() -> pd.DataFrame:
    """Retorna os baselines publicados no notebook sobre o periodo completo."""
    return pd.DataFrame(
        BASELINE_RESULTS,
        columns=["horizon", "reference", "mae_mw", "rmse_mw", "mape_pct"],
    )


def test_baseline_metrics_frame() -> pd.DataFrame:
    """Retorna os baselines recalculados na mesma particao de teste final."""
    return pd.DataFrame(
        TEST_BASELINE_RESULTS,
        columns=["horizon", "reference", "mae_mw", "rmse_mw", "mape_pct"],
    )


def validation_metrics_frame() -> pd.DataFrame:
    """Retorna o melhor conjunto de features por familia na validacao."""
    return pd.DataFrame(
        VALIDATION_RESULTS,
        columns=[
            "horizon",
            "model",
            "feature_set",
            "mae_mw",
            "rmse_mw",
            "mape_pct",
        ],
    )


def tuning_metrics_frame() -> pd.DataFrame:
    """Retorna a trilha do XGBoost selecionado ate o teste final."""
    frame = pd.DataFrame(
        TUNING_RESULTS,
        columns=[
            "horizon",
            "stage",
            "split",
            "mae_mw",
            "rmse_mw",
            "mape_pct",
        ],
    )
    frame["stage"] = pd.Categorical(
        frame["stage"],
        categories=[
            "XGBoost selecionado",
            "Early stopping",
            "XGBoost tunado",
            "Teste final",
        ],
        ordered=True,
    )
    return frame.sort_values(["horizon", "stage"]).reset_index(drop=True)


def worst_cases_frame() -> pd.DataFrame:
    """Retorna o maior erro absoluto observado em cada horizonte de teste."""
    frame = pd.DataFrame(
        WORST_CASES,
        columns=[
            "horizon",
            "timestamp",
            "actual_load_mw",
            "predicted_load_mw",
            "absolute_error_mw",
            "direction",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def best_baseline_comparison_frame() -> pd.DataFrame:
    """Compara o MAE final ao menor MAE de baseline de cada horizonte."""
    baselines = test_baseline_metrics_frame()
    indices = baselines.groupby("horizon", sort=False)["mae_mw"].idxmin()
    best = baselines.loc[indices, ["horizon", "reference", "mae_mw"]].rename(
        columns={"reference": "best_baseline", "mae_mw": "baseline_mae_mw"}
    )
    final = test_metrics_frame().loc[:, ["horizon", "mae_mw"]].rename(
        columns={"mae_mw": "model_mae_mw"}
    )
    result = final.merge(best, on="horizon", validate="one_to_one")
    result["mae_reduction_pct"] = (
        (result["baseline_mae_mw"] - result["model_mae_mw"])
        / result["baseline_mae_mw"]
        * 100
    )
    return result


def model_configuration_frame() -> pd.DataFrame:
    """Le features e hiperparametros realmente persistidos para treinamento."""
    rows = []
    for spec in load_model_specs():
        rows.append(
            {
                "horizon": spec.horizon,
                "feature_count": len(spec.features),
                "n_estimators": spec.params["n_estimators"],
                "learning_rate": spec.params["learning_rate"],
                "max_depth": spec.params["max_depth"],
                "min_child_weight": spec.params["min_child_weight"],
                "subsample": spec.params["subsample"],
                "colsample_bytree": spec.params["colsample_bytree"],
                "gamma": spec.params["gamma"],
                "reg_alpha": spec.params["reg_alpha"],
                "reg_lambda": spec.params["reg_lambda"],
            }
        )
    return pd.DataFrame(rows)
