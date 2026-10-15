"""Formatacao consistente de numeros e datas para a interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def as_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def format_number_br(value: Any, decimals: int = 0, signed: bool = False) -> str:
    number = as_finite_float(value)
    if number is None:
        return "Indisponível"
    sign = "+" if signed and number > 0 else ""
    formatted = f"{abs(number) if signed and number < 0 else number:,.{decimals}f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    if signed and number < 0:
        return f"-{formatted}"
    return f"{sign}{formatted}"


def format_mw(value: Any, decimals: int = 0, signed: bool = False) -> str:
    formatted = format_number_br(value, decimals=decimals, signed=signed)
    return formatted if formatted == "Indisponível" else f"{formatted} MW"


def format_percent(value: Any, decimals: int = 1, signed: bool = False) -> str:
    formatted = format_number_br(value, decimals=decimals, signed=signed)
    return formatted if formatted == "Indisponível" else f"{formatted}%"


def format_datetime(value: Any) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "Indisponível"
    if pd.isna(timestamp):
        return "Indisponível"
    return timestamp.strftime("%d/%m/%Y %H:%M")


def trend_symbol(value: Any) -> str:
    number = as_finite_float(value)
    if number is None or np.isclose(number, 0.0):
        return "→"
    return "↑" if number > 0 else "↓"
