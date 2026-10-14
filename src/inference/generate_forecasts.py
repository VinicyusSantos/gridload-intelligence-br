"""Gera as previsoes mais recentes de carga para 1h, 6h e 24h."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import (  # noqa: E402
    engineer_inference_features,
    read_base,
)
from src.models.train_models import (  # noqa: E402
    INPUT as TRAINING_INPUT,
    METADATA_FILE as TRAINING_METADATA_FILE,
    ModelSpec,
    load_model_specs,
    read_json,
)


INPUT = PROJECT_ROOT / "data/processed/base.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/predictions"
OUTPUT_PARQUET = OUTPUT_DIR / "latest_forecasts.parquet"
OUTPUT_JSON = OUTPUT_DIR / "latest_forecasts.json"

ModelLoader = Callable[[Path], XGBRegressor]


def horizon_hours(horizon: str) -> int:
    """Converte horizontes como ``1h`` e ``24h`` para horas inteiras."""
    if not horizon.endswith("h") or not horizon[:-1].isdigit():
        raise ValueError(f"Horizonte invalido: {horizon!r}")
    hours = int(horizon[:-1])
    if hours <= 0:
        raise ValueError(f"Horizonte invalido: {horizon!r}")
    return hours


def validate_inference_frame(
    frame: pd.DataFrame,
    specs: tuple[ModelSpec, ...],
) -> None:
    """Valida a base de features antes de carregar os modelos."""
    if frame.empty:
        raise ValueError("A base de inferencia esta vazia")
    if "timestamp" not in frame.columns:
        raise ValueError("A base de inferencia nao possui a coluna timestamp")

    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("A base de inferencia possui timestamps ausentes")
    if timestamps.duplicated().any():
        raise ValueError("A base de inferencia possui timestamps duplicados")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("A base de inferencia nao esta em ordem temporal")

    required_features = {
        feature for spec in specs for feature in spec.features
    }
    missing_features = required_features - set(frame.columns)
    if missing_features:
        raise ValueError(
            "Features ausentes na base de inferencia: "
            + ", ".join(sorted(missing_features))
        )

    latest = frame.iloc[-1]
    features_with_nulls = [
        feature
        for feature in sorted(required_features)
        if pd.isna(latest[feature])
    ]
    if features_with_nulls:
        raise ValueError(
            "Features ausentes na ultima observacao: "
            + ", ".join(features_with_nulls)
        )

    non_numeric_features = [
        feature
        for feature in sorted(required_features)
        if not is_numeric_dtype(frame[feature])
    ]
    if non_numeric_features:
        raise ValueError(
            "Features de inferencia nao numericas: "
            + ", ".join(non_numeric_features)
        )


def load_persisted_model(path: Path) -> XGBRegressor:
    """Carrega um modelo XGBoost persistido pelo treinamento."""
    if not path.is_file():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {path}")
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def generate_forecasts(
    feature_frame: pd.DataFrame,
    specs: tuple[ModelSpec, ...],
    training_metadata: dict[str, Any],
    *,
    model_loader: ModelLoader = load_persisted_model,
    generated_at: datetime | None = None,
) -> pd.DataFrame:
    """Aplica cada modelo a ultima observacao disponivel."""
    validate_inference_frame(feature_frame, specs)
    latest = feature_frame.iloc[[-1]]
    forecast_origin = pd.Timestamp(latest["timestamp"].iloc[0])
    generated_at = generated_at or datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        model = model_loader(spec.output_path)
        model_input = latest.loc[:, list(spec.features)]
        raw_prediction = np.asarray(model.predict(model_input)).reshape(-1)
        if raw_prediction.size != 1 or not np.isfinite(raw_prediction[0]):
            raise ValueError(
                f"O modelo {spec.horizon} retornou uma previsao invalida"
            )

        hours = horizon_hours(spec.horizon)
        rows.append(
            {
                "generated_at_utc": generated_at.isoformat(),
                "forecast_origin": forecast_origin,
                "horizon": spec.horizon,
                "horizon_hours": hours,
                "forecast_timestamp": forecast_origin
                + pd.Timedelta(hours=hours),
                "predicted_load_mw": float(raw_prediction[0]),
                "model_file": _display_path(spec.output_path),
                "trained_at_utc": training_metadata.get("trained_at_utc"),
                "training_rows": training_metadata.get("training_rows"),
                "training_period_end": training_metadata.get("period_end"),
            }
        )

    return pd.DataFrame(rows)


def save_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Salva as previsoes sem expor um parquet parcialmente gravado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_manifest_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Salva uma representacao JSON legivel das previsoes mais recentes."""
    forecasts = []
    for row in frame.to_dict(orient="records"):
        forecasts.append(
            {
                **row,
                "forecast_origin": pd.Timestamp(row["forecast_origin"]).isoformat(),
                "forecast_timestamp": pd.Timestamp(
                    row["forecast_timestamp"]
                ).isoformat(),
            }
        )

    payload = {
        "generated_at_utc": frame["generated_at_utc"].iloc[0],
        "forecast_origin": pd.Timestamp(
            frame["forecast_origin"].iloc[0]
        ).isoformat(),
        "feature_source": _display_path(INPUT),
        "training_source": _display_path(TRAINING_INPUT),
        "training_metadata": _display_path(TRAINING_METADATA_FILE),
        "forecasts": forecasts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    base = read_base(INPUT)
    feature_frame = engineer_inference_features(base)
    specs = load_model_specs()
    training_metadata = read_json(TRAINING_METADATA_FILE)
    forecasts = generate_forecasts(feature_frame, specs, training_metadata)
    save_parquet_atomic(forecasts, OUTPUT_PARQUET)
    save_manifest_atomic(forecasts, OUTPUT_JSON)

    print(
        "Origem das previsoes: "
        f"{pd.Timestamp(forecasts['forecast_origin'].iloc[0])}"
    )
    for row in forecasts.itertuples(index=False):
        print(
            f"{row.horizon}: {row.forecast_timestamp} -> "
            f"{row.predicted_load_mw:,.2f} MW"
        )
    print(f"Previsoes salvas: {_display_path(OUTPUT_PARQUET)}")
    print(f"Manifesto salvo: {_display_path(OUTPUT_JSON)}")


if __name__ == "__main__":
    main()
