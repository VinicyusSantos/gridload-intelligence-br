"""Limpeza e consolidacao das bases intermediarias da ONS e do INMET.

Este modulo concentra as etapas reutilizaveis desenvolvidas no notebook
``02_data_cleaning.ipynb`` e gera ``data/processed/base.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONS_INPUT = PROJECT_ROOT / "data/interim/load_clean.parquet"
INMET_INPUT = PROJECT_ROOT / "data/interim/weather_clean.parquet"
OUTPUT = PROJECT_ROOT / "data/processed/base.parquet"

ONS_REQUIRED_COLUMNS = {
    "timestamp",
    "subsystem_code",
    "subsystem",
    "load_mw",
}
INMET_REQUIRED_COLUMNS = {
    "timestamp",
    "region",
    "global_radiation_kj_m2",
}


def read_parquet(path: Path, source_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Base {source_name} nao encontrada: {path}")
    return pd.read_parquet(path)


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    source_name: str,
) -> None:
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Colunas ausentes na base {source_name}: "
            + ", ".join(sorted(missing_columns))
        )


def require_unique_timestamps(frame: pd.DataFrame, source_name: str) -> None:
    duplicated = frame["timestamp"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"A base {source_name} possui {int(duplicated.sum())} "
            "registros com timestamp duplicado"
        )


def prepare_ons(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ONS_REQUIRED_COLUMNS, "ONS")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")

    if result["timestamp"].dt.tz is not None:
        raise ValueError(
            "A base ONS passou a possuir timezone. Revise a regra de alinhamento "
            "temporal antes de continuar."
        )

    result = result.sort_values("timestamp").reset_index(drop=True)
    require_unique_timestamps(result, "ONS")

    if result["load_mw"].isna().any():
        raise ValueError("A base ONS possui valores ausentes em load_mw")
    if (result["load_mw"] < 0).any():
        raise ValueError("A base ONS possui valores negativos em load_mw")

    return result


def prepare_inmet(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, INMET_REQUIRED_COLUMNS, "INMET")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")

    # O INMET fornece a hora em UTC. A base atual da ONS possui timestamps sem
    # timezone; portanto, repetimos de forma explicita a premissa adotada no
    # notebook e removemos apenas a informacao de fuso, preservando a hora UTC.
    if result["timestamp"].dt.tz is not None:
        result["timestamp"] = (
            result["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        )

    result = result.sort_values("timestamp").reset_index(drop=True)
    require_unique_timestamps(result, "INMET")

    result["global_radiation_kj_m2"] = result[
        "global_radiation_kj_m2"
    ].interpolate(method="linear")

    if result["global_radiation_kj_m2"].isna().any():
        raise ValueError(
            "A interpolacao nao resolveu todos os valores ausentes de "
            "global_radiation_kj_m2"
        )

    return result


def merge_sources(ons: pd.DataFrame, inmet: pd.DataFrame) -> pd.DataFrame:
    processed = pd.merge(
        ons,
        inmet,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if processed.empty:
        raise ValueError("A juncao ONS + INMET nao produziu registros")

    processed = processed.drop(columns=["region", "subsystem"], errors="ignore")
    processed = processed.sort_values("timestamp").reset_index(drop=True)

    if processed.isna().any().any():
        missing = processed.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"A base processada ainda possui valores ausentes: {missing}")

    return processed


def save_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)


def main() -> None:
    ons = prepare_ons(read_parquet(ONS_INPUT, "ONS"))
    inmet = prepare_inmet(read_parquet(INMET_INPUT, "INMET"))
    processed = merge_sources(ons, inmet)
    save_parquet(processed, OUTPUT)

    print(f"Registros ONS: {len(ons):,}")
    print(f"Registros INMET: {len(inmet):,}")
    print(f"Registros processados: {len(processed):,}")
    print(f"Periodo: {processed['timestamp'].min()} ate {processed['timestamp'].max()}")
    print(f"Colunas: {len(processed.columns)}")
    print(f"Arquivo criado: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
