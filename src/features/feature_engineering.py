"""Cria as variaveis de modelagem a partir da base processada.

Este modulo concentra as etapas desenvolvidas no notebook
``04_feature_engineering.ipynb`` e gera
``data/processed/base_final.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data/processed/base.parquet"
OUTPUT = PROJECT_ROOT / "data/processed/base_final.parquet"

NORTHEAST_STATES = (
    "AL",
    "BA",
    "CE",
    "MA",
    "PB",
    "PE",
    "PI",
    "RN",
    "SE",
)

LOAD_LAGS = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS = (3, 6, 24)
FORECAST_HORIZONS = (1, 6, 24)
REQUIRED_COLUMNS = {"timestamp", "load_mw"}


def read_base(path: Path = INPUT) -> pd.DataFrame:
    """Le e valida a existencia da base processada."""
    if not path.is_file():
        raise FileNotFoundError(f"Base processada nao encontrada: {path}")
    return pd.read_parquet(path)


def prepare_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida a serie temporal usada pelas features defasadas."""
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Colunas ausentes na base processada: "
            + ", ".join(sorted(missing_columns))
        )

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result = result.sort_values("timestamp").reset_index(drop=True)

    if result.empty:
        raise ValueError("A base processada esta vazia")
    if result["timestamp"].isna().any():
        raise ValueError("A base processada possui timestamps ausentes")
    if result["timestamp"].duplicated().any():
        raise ValueError("A base processada possui timestamps duplicados")
    if result["load_mw"].isna().any():
        raise ValueError("A base processada possui valores ausentes em load_mw")

    intervals = result["timestamp"].diff().dropna()
    irregular_intervals = intervals.ne(pd.Timedelta(hours=1))
    if irregular_intervals.any():
        raise ValueError(
            "A base processada nao possui frequencia horaria continua; "
            f"foram encontrados {int(irregular_intervals.sum())} intervalos irregulares"
        )

    return result


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona calendario civil e feriados nacionais/estaduais."""
    result = frame.copy()
    timestamp = result["timestamp"]

    result["month"] = timestamp.dt.month
    result["weekday"] = timestamp.dt.day_name()
    result["weekend"] = timestamp.dt.dayofweek >= 5
    result["hour"] = timestamp.dt.hour
    result["weekday_num"] = timestamp.dt.weekday

    years = sorted(timestamp.dt.year.unique().tolist())
    national_holidays = holidays.country_holidays("BR", years=years)
    state_holidays = {
        state: holidays.country_holidays("BR", subdiv=state, years=years)
        for state in NORTHEAST_STATES
    }

    dates = timestamp.dt.date
    result["is_holiday_national"] = dates.isin(national_holidays).astype(int)

    def count_states_on_holiday(date: object) -> int:
        if date in national_holidays:
            return 0
        return sum(date in calendar for calendar in state_holidays.values())

    result["states_on_holiday"] = dates.map(count_states_on_holiday)
    return result


def add_cyclical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Codifica hora, dia da semana e mes em componentes seno e cosseno."""
    result = frame.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * result["weekday_num"] / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["weekday_num"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    return result


def add_load_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona defasagens, estatisticas moveis e rampa da carga."""
    result = frame.copy()
    previous_load = result["load_mw"].shift(1)

    for lag in LOAD_LAGS:
        result[f"load_lag_{lag}h"] = result["load_mw"].shift(lag)

    for window in ROLLING_WINDOWS:
        result[f"load_mean_{window}h"] = previous_load.rolling(window).mean()

    for window in ROLLING_WINDOWS:
        result[f"load_std_{window}h"] = previous_load.rolling(window).std()

    result["ramp_1h"] = result["load_mw"].diff(1).shift(1)
    return result


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona os alvos futuros para os horizontes definidos no notebook."""
    result = frame.copy()
    for horizon in FORECAST_HORIZONS:
        result[f"target_load_{horizon}h"] = result["load_mw"].shift(-horizon)
    return result


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Executa toda a engenharia de features e remove bordas incompletas."""
    result = prepare_base(frame)
    result = add_calendar_features(result)
    result = add_cyclical_features(result)
    result = add_load_history_features(result)
    result = add_targets(result)
    return result.dropna().reset_index(drop=True)


def save_parquet(frame: pd.DataFrame, output_path: Path = OUTPUT) -> None:
    """Grava o resultado de forma atomica."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)


def main() -> None:
    base = read_base()
    featured = engineer_features(base)
    save_parquet(featured)

    print(f"Registros de entrada: {len(base):,}")
    print(f"Registros com features: {len(featured):,}")
    print(f"Periodo: {featured['timestamp'].min()} ate {featured['timestamp'].max()}")
    print(f"Colunas: {len(featured.columns)}")
    print(f"Arquivo criado: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
