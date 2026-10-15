"""Explicabilidade local e global dos modelos persistidos."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.dashboard.config import GLOBAL_SHAP_PATH, GLOBAL_SHAP_SAMPLE_SIZE
from src.dashboard.data_loader import load_base_data, load_featured_data, load_model
from src.features.feature_engineering import engineer_inference_features
from src.inference.generate_forecasts import save_parquet_atomic
from src.models.train_models import ModelSpec, load_model_specs


FEATURE_NAMES = {
    "hour_sin": "Horário do dia",
    "hour_cos": "Horário do dia",
    "weekday_sin": "Dia da semana",
    "weekday_cos": "Dia da semana",
    "month_sin": "Época do ano",
    "month_cos": "Época do ano",
    "weekend": "Fim de semana",
    "is_holiday_national": "Feriado nacional",
    "states_on_holiday": "Estados em feriado",
    "load_mw": "Carga atual",
    "load_lag_1h": "Carga há 1 hora",
    "load_lag_2h": "Carga há 2 horas",
    "load_lag_3h": "Carga há 3 horas",
    "load_lag_24h": "Carga há 24 horas",
    "load_lag_48h": "Carga há 48 horas",
    "load_lag_168h": "Carga há 7 dias",
    "load_mean_3h": "Média de carga das últimas 3h",
    "load_mean_6h": "Média de carga das últimas 6h",
    "load_mean_24h": "Média de carga das últimas 24h",
    "load_std_3h": "Variabilidade da carga em 3 horas",
    "load_std_6h": "Variabilidade da carga em 6 horas",
    "load_std_24h": "Variabilidade da carga em 24 horas",
    "ramp_1h": "Variação recente da carga",
    "precipitation_mm": "Precipitação",
    "pressure_station_hpa": "Pressão atmosférica",
    "global_radiation_kj_m2": "Radiação solar",
    "temperature_c": "Temperatura",
    "humidity_pct": "Umidade",
    "wind_gust_ms": "Rajada de vento",
    "wind_speed_ms": "Velocidade do vento",
}


def friendly_feature_name(feature: str) -> str:
    return FEATURE_NAMES.get(feature, feature.replace("_", " ").capitalize())


@st.cache_resource(show_spinner=False)
def _tree_explainer(
    model_path: str,
    modified_ns: int,
    size: int,
) -> shap.TreeExplainer:
    del modified_ns, size
    return shap.TreeExplainer(load_model(Path(model_path)))


def _get_explainer(spec: ModelSpec) -> shap.TreeExplainer:
    stat = spec.output_path.stat()
    return _tree_explainer(
        str(spec.output_path),
        stat.st_mtime_ns,
        stat.st_size,
    )


def _shap_matrix(explainer: shap.TreeExplainer, features: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(features)
    if isinstance(values, list):
        values = values[0]
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape != features.shape:
        raise ValueError(
            "As dimensoes SHAP nao correspondem as features usadas pelo modelo."
        )
    return matrix


def group_shap_values(
    values: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Agrupa componentes ciclicas e devolve uma coluna por conceito."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    grouped: dict[str, np.ndarray] = {}
    for index, feature in enumerate(feature_names):
        label = friendly_feature_name(feature)
        contribution = matrix[:, index]
        grouped[label] = grouped.get(label, np.zeros(matrix.shape[0])) + contribution
    return pd.DataFrame(grouped)


def _spec_for_horizon(horizon: str) -> ModelSpec:
    for spec in load_model_specs():
        if spec.horizon == horizon:
            return spec
    raise ValueError(f"Horizonte desconhecido: {horizon}")


def local_explanation(horizon: str) -> pd.DataFrame:
    """Calcula SHAP da mesma linha usada na previsao mais recente."""
    spec = _spec_for_horizon(horizon)
    features = engineer_inference_features(load_base_data()).iloc[[-1]]
    model_input = features.loc[:, list(spec.features)]
    values = _shap_matrix(_get_explainer(spec), model_input)
    grouped = group_shap_values(values, spec.features).iloc[0]
    result = grouped.rename_axis("feature").reset_index(name="impact_mw")
    result["abs_impact_mw"] = result["impact_mw"].abs()
    result["direction"] = np.where(
        result["impact_mw"] >= 0,
        "Eleva a previsão",
        "Reduz a previsão",
    )
    return result.sort_values("abs_impact_mw", ascending=False).reset_index(drop=True)


def build_global_shap_artifact(
    *,
    output_path: Path = GLOBAL_SHAP_PATH,
    sample_size: int = GLOBAL_SHAP_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Calcula importancia SHAP global real e persiste um artefato leve."""
    featured = load_featured_data()
    if featured.empty:
        raise ValueError("A base de features esta vazia.")
    n_rows = min(int(sample_size), len(featured))
    if n_rows <= 0:
        raise ValueError("sample_size deve ser positivo.")
    positions = np.linspace(0, len(featured) - 1, num=n_rows, dtype=int)
    sample = featured.iloc[positions]
    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    for spec in load_model_specs():
        model_input = sample.loc[:, list(spec.features)]
        values = _shap_matrix(_get_explainer(spec), model_input)
        grouped = group_shap_values(values, spec.features)
        for feature, mean_abs in grouped.abs().mean().items():
            records.append(
                {
                    "horizon": spec.horizon,
                    "feature": feature,
                    "mean_abs_shap_mw": float(mean_abs),
                    "sample_size": n_rows,
                    "generated_at_utc": generated_at,
                }
            )

    artifact = pd.DataFrame(records).sort_values(
        ["horizon", "mean_abs_shap_mw"],
        ascending=[True, False],
    )
    save_parquet_atomic(artifact, output_path)
    return artifact.reset_index(drop=True)
