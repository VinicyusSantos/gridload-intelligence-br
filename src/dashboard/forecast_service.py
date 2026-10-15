"""Servico central de previsao consumido por todas as paginas do dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.context.historical_percentile import add_historical_context_to_forecasts
from src.dashboard.config import FORECASTS_JSON_PATH, FORECASTS_PATH, HORIZONS
from src.dashboard.data_loader import (
    DashboardDataError,
    clear_data_caches,
    load_base_data,
    load_latest_forecasts,
    load_model,
)
from src.dashboard.formatting import format_mw
from src.features.feature_engineering import engineer_inference_features
from src.inference.generate_forecasts import (
    generate_forecasts,
    save_manifest_atomic,
    save_parquet_atomic,
)
from src.models.train_models import METADATA_FILE, load_model_specs, read_json


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if pd.isna(timestamp) else timestamp


def _artifact_is_current(
    forecasts: pd.DataFrame,
    base: pd.DataFrame,
    training_metadata: dict[str, Any],
) -> bool:
    required = {
        "forecast_origin",
        "horizon",
        "predicted_load_mw",
        "trained_at_utc",
    }
    if forecasts.empty or not required.issubset(forecasts.columns):
        return False
    if set(forecasts["horizon"].astype(str)) != set(HORIZONS):
        return False

    base_end = pd.to_datetime(base["timestamp"], errors="coerce").max()
    origins = pd.to_datetime(forecasts["forecast_origin"], errors="coerce")
    if pd.isna(base_end) or origins.isna().any() or not origins.eq(base_end).all():
        return False

    trained_at = training_metadata.get("trained_at_utc")
    if not trained_at:
        return False
    return forecasts["trained_at_utc"].fillna("").eq(str(trained_at)).all()


def _calculate_and_persist(
    base: pd.DataFrame,
    training_metadata: dict[str, Any],
) -> pd.DataFrame:
    feature_frame = engineer_inference_features(base)
    forecasts = generate_forecasts(
        feature_frame,
        load_model_specs(),
        training_metadata,
        model_loader=load_model,
        generated_at=datetime.now(timezone.utc),
    )
    forecasts = add_historical_context_to_forecasts(
        history=base,
        forecasts=forecasts,
        min_samples=30,
        use_holiday=False,
    )
    save_parquet_atomic(forecasts, FORECASTS_PATH)
    save_manifest_atomic(forecasts, FORECASTS_JSON_PATH)
    clear_data_caches()
    return forecasts


def _load_or_calculate_forecasts(
    base: pd.DataFrame,
    training_metadata: dict[str, Any],
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, bool]:
    if not refresh:
        try:
            forecasts = load_latest_forecasts()
        except DashboardDataError:
            forecasts = pd.DataFrame()
        if _artifact_is_current(forecasts, base, training_metadata):
            return forecasts, False
    return _calculate_and_persist(base, training_metadata), True


def _forecast_row(row: pd.Series, current_load: float) -> dict[str, Any]:
    prediction = float(row["predicted_load_mw"])
    delta_mw = prediction - current_load
    delta_pct = (delta_mw / current_load * 100) if current_load else None
    percentile = _finite_or_none(row.get("historical_percentile"))
    sample_size = _finite_or_none(row.get("historical_sample_size"))

    return {
        "horizon": str(row["horizon"]),
        "forecast_mw": prediction,
        "target_timestamp": _as_timestamp(row["forecast_timestamp"]),
        "delta_mw": delta_mw,
        "delta_pct": delta_pct,
        "percentile": percentile,
        "percentile_label": row.get("historical_percentile_label"),
        "historical_sample_size": int(sample_size) if sample_size is not None else 0,
        "historical_reference": row.get("historical_reference"),
        "historical_interpretation": row.get("historical_interpretation"),
        "historical_context_status": row.get("historical_context_status"),
        "historical_context_reason": row.get("historical_context_reason"),
        "trained_at_utc": row.get("trained_at_utc"),
    }


def get_current_forecast(*, refresh: bool = False) -> dict[str, Any]:
    """Retorna o snapshot atual; esta e a unica entrada de previsao da UI.

    ``refresh=True`` refaz somente a inferencia com os modelos persistidos. A
    funcao nunca executa ingestao, engenharia da base de treino ou treinamento.
    """
    base = load_base_data()
    if base.empty or not {"timestamp", "load_mw"}.issubset(base.columns):
        raise DashboardDataError(
            "A base processada esta vazia ou nao possui timestamp e load_mw."
        )

    base = base.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="raise")
    base = base.sort_values("timestamp").reset_index(drop=True)
    training_metadata = read_json(METADATA_FILE)
    forecasts, recalculated = _load_or_calculate_forecasts(
        base,
        training_metadata,
        refresh=refresh,
    )

    current_load = float(base["load_mw"].iloc[-1])
    previous_load = float(base["load_mw"].iloc[-2]) if len(base) > 1 else None
    current_delta_mw = (
        current_load - previous_load if previous_load is not None else None
    )
    current_delta_pct = (
        current_delta_mw / previous_load * 100
        if previous_load not in (None, 0)
        else None
    )

    rows = {
        str(row["horizon"]): _forecast_row(row, current_load)
        for _, row in forecasts.iterrows()
    }
    missing = set(HORIZONS) - set(rows)
    if missing:
        raise DashboardDataError(
            "Horizontes ausentes no artefato de previsao: "
            + ", ".join(sorted(missing))
        )

    generated_at = _as_timestamp(forecasts["generated_at_utc"].iloc[0])
    forecast_origin = _as_timestamp(forecasts["forecast_origin"].iloc[0])
    return {
        "generated_at": generated_at,
        "forecast_origin": forecast_origin,
        "current_timestamp": pd.Timestamp(base["timestamp"].iloc[-1]),
        "current_load_mw": current_load,
        "previous_load_mw": previous_load,
        "current_delta_mw": current_delta_mw,
        "current_delta_pct": current_delta_pct,
        "forecasts": rows,
        "recalculated": recalculated,
    }


def forecast_summary(snapshot: dict[str, Any]) -> str:
    """Cria um resumo deterministico, sem texto gerado ou dados inventados."""
    forecasts = snapshot["forecasts"]
    deltas = [forecasts[horizon]["delta_mw"] for horizon in HORIZONS]
    directions = [np.sign(delta) for delta in deltas]

    if all(direction > 0 for direction in directions):
        movement = "Os três horizontes indicam carga acima do nível atual."
    elif all(direction < 0 for direction in directions):
        movement = "Os três horizontes indicam carga abaixo do nível atual."
    else:
        movement = "Os horizontes apontam movimentos diferentes em relação ao nível atual."

    highest_horizon = max(
        HORIZONS,
        key=lambda horizon: forecasts[horizon]["forecast_mw"],
    )
    contextual = [
        forecasts[horizon]
        for horizon in HORIZONS
        if forecasts[horizon]["historical_context_status"] == "available"
        and forecasts[horizon]["percentile"] is not None
    ]
    context_text = ""
    if contextual:
        extreme = max(contextual, key=lambda item: item["percentile"])
        context_text = (
            f" O maior posicionamento histórico aparece em {extreme['horizon']} "
            f"({extreme['percentile_label']})."
        )

    return (
        f"A carga atual é de {format_mw(snapshot['current_load_mw'])}. "
        f"{movement} O maior valor previsto é o de {highest_horizon}, "
        f"com {format_mw(forecasts[highest_horizon]['forecast_mw'])}."
        f"{context_text}"
    )
