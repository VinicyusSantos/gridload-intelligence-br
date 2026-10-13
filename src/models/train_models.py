"""Treina os modelos XGBoost de 1h, 6h e 24h com a base mais recente.

Os hiperparametros e as listas de features foram definidos no notebook
``06_forecasting.ipynb``. O pipeline reutiliza essas configuracoes e ajusta
novamente os tres modelos sobre todos os registros rotulados produzidos pela
etapa de engenharia de features.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from xgboost import XGBRegressor, __version__ as XGBOOST_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data/processed/base_final.parquet"
MODELS_DIR = PROJECT_ROOT / "src/models"
PARAMS_FILE = MODELS_DIR / "params.json"
FEATURES_FILE = MODELS_DIR / "features.json"
METADATA_FILE = MODELS_DIR / "training_metadata.json"

HORIZONS = ("1h", "6h", "24h")
REQUIRED_PARAMS = {
    "objective",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "random_state",
    "n_jobs",
}


@dataclass(frozen=True)
class ModelSpec:
    """Configuracao necessaria para treinar um horizonte de previsao."""

    horizon: str
    features: tuple[str, ...]
    target: str
    params: dict[str, Any]
    output_path: Path


def read_json(path: Path) -> dict[str, Any]:
    """Le um arquivo JSON de configuracao e valida seu formato basico."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuracao nao encontrada: {path}")

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"A configuracao deve ser um objeto JSON: {path}")
    return payload


def _validate_model_spec(spec: ModelSpec) -> None:
    if not spec.features:
        raise ValueError(f"O modelo {spec.horizon} nao possui features")
    if len(spec.features) != len(set(spec.features)):
        raise ValueError(f"O modelo {spec.horizon} possui features duplicadas")

    missing_params = REQUIRED_PARAMS - set(spec.params)
    if missing_params:
        raise ValueError(
            f"Hiperparametros ausentes para {spec.horizon}: "
            + ", ".join(sorted(missing_params))
        )

    n_estimators = spec.params["n_estimators"]
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, int)
        or n_estimators <= 0
    ):
        raise ValueError(
            f"n_estimators invalido para {spec.horizon}: {n_estimators!r}"
        )


def load_model_specs(
    params_path: Path = PARAMS_FILE,
    features_path: Path = FEATURES_FILE,
    models_dir: Path = MODELS_DIR,
) -> tuple[ModelSpec, ...]:
    """Combina os parametros e features persistidos para os tres horizontes."""
    params_payload = read_json(params_path)
    features_payload = read_json(features_path)
    specs: list[ModelSpec] = []

    for horizon in HORIZONS:
        params_key = f"best_params_{horizon}"
        features_key = f"features_{horizon}"

        params = params_payload.get(params_key)
        features = features_payload.get(features_key)
        if not isinstance(params, dict):
            raise ValueError(f"Configuracao ausente ou invalida: {params_key}")
        if not isinstance(features, list) or not all(
            isinstance(feature, str) for feature in features
        ):
            raise ValueError(f"Configuracao ausente ou invalida: {features_key}")

        spec = ModelSpec(
            horizon=horizon,
            features=tuple(features),
            target=f"target_load_{horizon}",
            params=dict(params),
            output_path=models_dir / f"xgb_{horizon}.json",
        )
        _validate_model_spec(spec)
        specs.append(spec)

    return tuple(specs)


def read_training_frame(path: Path = INPUT) -> pd.DataFrame:
    """Le a base com features que sera usada no reajuste dos modelos."""
    if not path.is_file():
        raise FileNotFoundError(f"Base de treinamento nao encontrada: {path}")
    return pd.read_parquet(path)


def validate_training_frame(
    frame: pd.DataFrame,
    specs: tuple[ModelSpec, ...],
) -> None:
    """Valida dados, ordenacao temporal e colunas exigidas pelos modelos."""
    if frame.empty:
        raise ValueError("A base de treinamento esta vazia")
    if "timestamp" not in frame.columns:
        raise ValueError("A base de treinamento nao possui a coluna timestamp")

    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("A base de treinamento possui timestamps ausentes")
    if timestamps.duplicated().any():
        raise ValueError("A base de treinamento possui timestamps duplicados")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("A base de treinamento nao esta em ordem temporal")

    required_columns = {
        column
        for spec in specs
        for column in (*spec.features, spec.target)
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Colunas ausentes na base de treinamento: "
            + ", ".join(sorted(missing_columns))
        )

    columns_with_nulls = [
        column for column in sorted(required_columns) if frame[column].isna().any()
    ]
    if columns_with_nulls:
        raise ValueError(
            "Colunas de treinamento com valores ausentes: "
            + ", ".join(columns_with_nulls)
        )

    non_numeric_columns = [
        column
        for column in sorted(required_columns)
        if not is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            "Colunas de treinamento nao numericas: "
            + ", ".join(non_numeric_columns)
        )


def train_model(frame: pd.DataFrame, spec: ModelSpec) -> XGBRegressor:
    """Ajusta um modelo usando todos os registros rotulados disponiveis."""
    model = XGBRegressor(**spec.params)
    model.fit(frame.loc[:, list(spec.features)], frame.loc[:, spec.target])
    return model


def save_model_atomic(model: XGBRegressor, output_path: Path) -> None:
    """Persiste um modelo sem expor um arquivo parcialmente gravado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}-",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        model.save_model(str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Grava metadados JSON de forma atomica."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def build_training_metadata(
    frame: pd.DataFrame,
    specs: tuple[ModelSpec, ...],
    input_path: Path,
) -> dict[str, Any]:
    """Cria o manifesto que permite auditar cada execucao de treinamento."""
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": "notebooks/06_forecasting.ipynb",
        "input_file": _display_path(input_path),
        "training_rows": len(frame),
        "period_start": timestamps.min().isoformat(),
        "period_end": timestamps.max().isoformat(),
        "xgboost_version": XGBOOST_VERSION,
        "models": {
            spec.horizon: {
                "target": spec.target,
                "feature_count": len(spec.features),
                "features": list(spec.features),
                "params": spec.params,
                "model_file": _display_path(spec.output_path),
            }
            for spec in specs
        },
    }


def train_and_save_models(
    frame: pd.DataFrame,
    specs: tuple[ModelSpec, ...],
    input_path: Path = INPUT,
    metadata_path: Path = METADATA_FILE,
) -> tuple[dict[str, XGBRegressor], dict[str, Any]]:
    """Treina todos os horizontes e atualiza seus artefatos persistidos."""
    validate_training_frame(frame, specs)
    trained_models: dict[str, XGBRegressor] = {}

    # Todos os modelos sao ajustados antes de qualquer artefato existente ser
    # substituido. Assim, uma falha de treinamento nao deixa apenas parte dos
    # horizontes atualizada.
    for spec in specs:
        print(
            f"Treinando modelo {spec.horizon}: {len(frame):,} registros, "
            f"{len(spec.features)} features, "
            f"{spec.params['n_estimators']:,} arvores...",
            flush=True,
        )
        trained_models[spec.horizon] = train_model(frame, spec)

    for spec in specs:
        save_model_atomic(trained_models[spec.horizon], spec.output_path)
        print(f"Modelo salvo: {_display_path(spec.output_path)}", flush=True)

    metadata = build_training_metadata(frame, specs, input_path)
    write_json_atomic(metadata, metadata_path)
    print(f"Metadados salvos: {_display_path(metadata_path)}", flush=True)
    return trained_models, metadata


def main() -> None:
    specs = load_model_specs()
    frame = read_training_frame()
    _, metadata = train_and_save_models(frame, specs)

    print(f"Registros usados no treinamento: {metadata['training_rows']:,}")
    print(
        f"Periodo rotulado: {metadata['period_start']} ate "
        f"{metadata['period_end']}"
    )


if __name__ == "__main__":
    main()
