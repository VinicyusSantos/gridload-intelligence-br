"""Leitura cacheada dos artefatos usados pelo Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

from src.dashboard.config import (
    BASE_PATH,
    FEATURED_BASE_PATH,
    FORECASTS_PATH,
    GLOBAL_SHAP_PATH,
    MODEL_COMPARISON_PATH,
    TEST_PREDICTIONS_PATH,
)


class DashboardDataError(RuntimeError):
    """Erro de dado ausente ou invalido que pode ser exibido na interface."""


def _signature(path: Path) -> tuple[str, int, int]:
    if not path.is_file():
        raise DashboardDataError(f"Arquivo nao encontrado: {path}")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns, size
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, modified_ns: int, size: int) -> dict[str, Any]:
    del modified_ns, size
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise DashboardDataError(f"JSON invalido: {path}")
    return payload


@st.cache_resource(show_spinner=False)
def _load_xgboost_model(
    path: str,
    modified_ns: int,
    size: int,
) -> XGBRegressor:
    del modified_ns, size
    model = XGBRegressor()
    model.load_model(path)
    return model


def load_base_data() -> pd.DataFrame:
    return _read_parquet(*_signature(BASE_PATH)).copy()


def load_featured_data() -> pd.DataFrame:
    return _read_parquet(*_signature(FEATURED_BASE_PATH)).copy()


def load_latest_forecasts() -> pd.DataFrame:
    return _read_parquet(*_signature(FORECASTS_PATH)).copy()


def load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return _read_parquet(*_signature(path)).copy()


def load_global_shap() -> pd.DataFrame | None:
    return load_optional_parquet(GLOBAL_SHAP_PATH)


def load_model_comparison() -> pd.DataFrame | None:
    return load_optional_parquet(MODEL_COMPARISON_PATH)


def load_test_predictions() -> pd.DataFrame | None:
    return load_optional_parquet(TEST_PREDICTIONS_PATH)


def load_json(path: Path) -> dict[str, Any]:
    return dict(_read_json(*_signature(path)))


def load_model(path: Path) -> XGBRegressor:
    return _load_xgboost_model(*_signature(path))


def clear_data_caches() -> None:
    _read_parquet.clear()
    _read_json.clear()
