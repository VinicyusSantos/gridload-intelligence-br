"""Contexto historico por dia da semana e hora para previsoes de carga."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


WEEKDAY_DESCRIPTIONS = {
    0: "segundas-feiras",
    1: "terças-feiras",
    2: "quartas-feiras",
    3: "quintas-feiras",
    4: "sextas-feiras",
    5: "sábados",
    6: "domingos",
}


def _timestamp_text(value: pd.Timestamp) -> str:
    return value.isoformat(sep=" ")


def _reference_description(weekday: int, hour: int) -> str:
    return f"{WEEKDAY_DESCRIPTIONS[weekday]} às {hour:02d}h"


def _base_result(
    prediction: Any,
    target_timestamp: Any,
) -> dict[str, Any]:
    return {
        "prediction_mw": prediction,
        "percentile": None,
        "percentile_label": None,
        "sample_size": 0,
        "target_timestamp": target_timestamp,
        "reference_weekday": None,
        "reference_hour": None,
        "reference_description": None,
        "reference_level": None,
        "interpretation": None,
        "status": "unavailable",
        "reason_code": None,
        "reason": None,
        "holiday_filter_requested": False,
        "holiday_filter_used": False,
        "target_is_holiday": None,
        "fallback_reason": None,
        "excluded_value_count": 0,
        "duplicate_timestamp_count": 0,
    }


def _unavailable(
    result: dict[str, Any],
    reason_code: str,
    reason: str,
    **updates: Any,
) -> dict[str, Any]:
    result.update(updates)
    result["status"] = "unavailable"
    result["reason_code"] = reason_code
    result["reason"] = reason
    return result


def _parse_prediction(prediction: Any) -> float | None:
    if isinstance(prediction, (bool, np.bool_)):
        return None
    try:
        value = float(prediction)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _parse_target_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if pd.isna(timestamp) else timestamp


def _coerce_holiday_value(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "sim", "yes"}:
            return True
        if normalized in {"false", "0", "nao", "no"}:
            return False
    return None


def _coerce_holiday_series(values: pd.Series) -> pd.Series:
    return values.map(_coerce_holiday_value).astype("boolean")


def historical_percentile(
    history: pd.DataFrame,
    prediction: Any,
    target_timestamp: Any,
    value_col: str = "load_mw",
    timestamp_col: str = "timestamp",
    min_samples: int = 30,
    use_holiday: bool = False,
    holiday_col: str = "is_holiday",
    target_is_holiday: Any = None,
) -> dict[str, Any]:
    """Calcula o percentil empirico em um grupo historicamente comparavel.

    A referencia principal e formada por observacoes com o mesmo dia da semana
    e hora do ``target_timestamp``. Somente linhas estritamente anteriores ao
    target entram no calculo, o que torna a funcao segura para backtesting.
    """
    if isinstance(min_samples, bool) or not isinstance(min_samples, int):
        raise ValueError("min_samples deve ser um inteiro positivo")
    if min_samples <= 0:
        raise ValueError("min_samples deve ser um inteiro positivo")

    result = _base_result(prediction, target_timestamp)
    result["holiday_filter_requested"] = bool(use_holiday)

    parsed_prediction = _parse_prediction(prediction)
    if parsed_prediction is None:
        return _unavailable(
            result,
            "invalid_prediction",
            "A previsao deve ser um numero finito.",
            prediction_mw=None,
        )
    result["prediction_mw"] = parsed_prediction

    target = _parse_target_timestamp(target_timestamp)
    if target is None:
        return _unavailable(
            result,
            "invalid_target_timestamp",
            "O target_timestamp e invalido.",
            target_timestamp=None,
        )

    target_weekday = int(target.weekday())
    target_hour = int(target.hour)
    description = _reference_description(target_weekday, target_hour)
    result.update(
        {
            "target_timestamp": _timestamp_text(target),
            "reference_weekday": target_weekday,
            "reference_hour": target_hour,
            "reference_description": description,
        }
    )

    if not isinstance(history, pd.DataFrame):
        return _unavailable(
            result,
            "invalid_history",
            "O historico deve ser um DataFrame do pandas.",
        )
    if history.empty:
        return _unavailable(
            result,
            "empty_history",
            "O historico esta vazio.",
        )

    missing_columns = {
        column
        for column in (timestamp_col, value_col)
        if column not in history.columns
    }
    if missing_columns:
        return _unavailable(
            result,
            "missing_history_columns",
            "Colunas ausentes no historico: "
            + ", ".join(sorted(missing_columns)),
        )

    try:
        history_timestamps = pd.to_datetime(
            history[timestamp_col],
            errors="coerce",
        )
    except (TypeError, ValueError, OverflowError):
        return _unavailable(
            result,
            "invalid_history_timestamps",
            "Os timestamps historicos nao puderam ser interpretados.",
        )

    invalid_timestamp_count = int(history_timestamps.isna().sum())
    if invalid_timestamp_count:
        return _unavailable(
            result,
            "invalid_history_timestamps",
            "O historico possui timestamps invalidos ou ausentes.",
            invalid_timestamp_count=invalid_timestamp_count,
        )

    try:
        history_timezone = history_timestamps.dt.tz
    except AttributeError:
        return _unavailable(
            result,
            "mixed_history_timezones",
            "O historico mistura timestamps com configuracoes de timezone diferentes.",
        )

    target_timezone = target.tz
    if (history_timezone is None) != (target_timezone is None):
        return _unavailable(
            result,
            "timezone_mismatch",
            "Historico e target_timestamp devem ser ambos naive ou ambos timezone-aware.",
        )

    if history_timezone is not None:
        history_timestamps = history_timestamps.dt.tz_convert(target_timezone)

    eligible = pd.DataFrame(
        {
            "timestamp": history_timestamps,
            "value": history[value_col],
        },
        index=history.index,
    )
    if holiday_col in history.columns:
        eligible["holiday"] = history[holiday_col]

    eligible = eligible.loc[eligible["timestamp"] < target].copy()

    duplicate_rows = eligible["timestamp"].duplicated(keep=False)
    if duplicate_rows.any():
        duplicate_count = int(
            eligible.loc[duplicate_rows, "timestamp"].nunique()
        )
        return _unavailable(
            result,
            "duplicate_historical_timestamps",
            "O historico anterior ao target possui timestamps duplicados; "
            "nenhuma linha foi removida automaticamente.",
            duplicate_timestamp_count=duplicate_count,
        )

    numeric_values = pd.to_numeric(eligible["value"], errors="coerce")
    numeric_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
    finite_values = pd.Series(
        np.isfinite(numeric_array),
        index=numeric_values.index,
    )
    excluded_value_count = int((~finite_values).sum())
    eligible = eligible.loc[finite_values].copy()
    eligible["value"] = numeric_values.loc[finite_values].astype(float)
    result["excluded_value_count"] = excluded_value_count

    eligible["weekday"] = eligible["timestamp"].dt.weekday
    eligible["hour"] = eligible["timestamp"].dt.hour
    comparable = eligible.loc[
        (eligible["weekday"] == target_weekday)
        & (eligible["hour"] == target_hour)
    ].copy()

    reference = comparable
    reference_level = "weekday_hour"
    fallback_reason: str | None = None
    parsed_target_holiday = _coerce_holiday_value(target_is_holiday)
    result["target_is_holiday"] = parsed_target_holiday

    if use_holiday:
        if holiday_col not in history.columns:
            fallback_reason = "holiday_column_missing"
        elif parsed_target_holiday is None:
            fallback_reason = "target_holiday_not_provided_or_invalid"
        else:
            comparable["holiday"] = _coerce_holiday_series(
                comparable["holiday"]
            )
            holiday_reference = comparable.loc[
                comparable["holiday"] == parsed_target_holiday
            ]
            if len(holiday_reference) >= min_samples:
                reference = holiday_reference
                reference_level = "weekday_hour_holiday"
                result["holiday_filter_used"] = True
                holiday_text = (
                    " em feriados"
                    if parsed_target_holiday
                    else " em dias não feriados"
                )
                description += holiday_text
            else:
                fallback_reason = "insufficient_holiday_samples"

    result["fallback_reason"] = fallback_reason
    result["reference_description"] = description
    result["reference_level"] = reference_level
    result["sample_size"] = int(len(reference))

    if reference.empty:
        return _unavailable(
            result,
            "no_comparable_history",
            "Nao existem observacoes historicas anteriores comparaveis.",
        )
    if len(reference) < min_samples:
        return _unavailable(
            result,
            "insufficient_samples",
            f"A referencia possui {len(reference)} observacoes; "
            f"o minimo exigido e {min_samples}.",
        )

    percentile = float((reference["value"] <= parsed_prediction).mean() * 100)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    percentile_label = int(np.floor(percentile + 0.5))
    rounded_percentile = round(percentile, 4)

    result.update(
        {
            "percentile": rounded_percentile,
            "percentile_label": f"P{percentile_label}",
            "interpretation": (
                "A previsão está acima de aproximadamente "
                f"{rounded_percentile:.1f}% das cargas historicamente "
                f"observadas em {description}."
            ),
            "status": "available",
            "reason_code": None,
            "reason": None,
        }
    )
    return result


def historical_percentiles_for_forecasts(
    history: pd.DataFrame,
    forecasts: Mapping[str, Any],
    target_timestamps: Mapping[str, Any],
    *,
    value_col: str = "load_mw",
    timestamp_col: str = "timestamp",
    min_samples: int = 30,
    use_holiday: bool = False,
    holiday_col: str = "is_holiday",
    target_holidays: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Calcula simultaneamente o contexto dos horizontes informados."""
    results: dict[str, dict[str, Any]] = {}
    target_holidays = target_holidays or {}

    for horizon, forecast in forecasts.items():
        context = historical_percentile(
            history=history,
            prediction=forecast,
            target_timestamp=target_timestamps.get(horizon),
            value_col=value_col,
            timestamp_col=timestamp_col,
            min_samples=min_samples,
            use_holiday=use_holiday,
            holiday_col=holiday_col,
            target_is_holiday=target_holidays.get(horizon),
        )
        results[horizon] = {
            "forecast_mw": context["prediction_mw"],
            **context,
        }

    return results


def add_historical_context_to_forecasts(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    prediction_col: str = "predicted_load_mw",
    target_col: str = "forecast_timestamp",
    value_col: str = "load_mw",
    timestamp_col: str = "timestamp",
    min_samples: int = 30,
    use_holiday: bool = False,
    holiday_col: str = "is_holiday",
    target_holiday_col: str | None = None,
) -> pd.DataFrame:
    """Adiciona o contexto historico a um DataFrame de forecasts."""
    required_columns = {prediction_col, target_col}
    missing_columns = required_columns - set(forecasts.columns)
    if missing_columns:
        raise ValueError(
            "Colunas ausentes nas previsoes: "
            + ", ".join(sorted(missing_columns))
        )

    enriched = forecasts.copy().reset_index(drop=True)
    contexts: list[dict[str, Any]] = []
    for row in enriched.to_dict(orient="records"):
        target_holiday = (
            row.get(target_holiday_col) if target_holiday_col else None
        )
        contexts.append(
            historical_percentile(
                history=history,
                prediction=row[prediction_col],
                target_timestamp=row[target_col],
                value_col=value_col,
                timestamp_col=timestamp_col,
                min_samples=min_samples,
                use_holiday=use_holiday,
                holiday_col=holiday_col,
                target_is_holiday=target_holiday,
            )
        )

    enriched["historical_percentile"] = [
        context["percentile"] for context in contexts
    ]
    enriched["historical_percentile_label"] = [
        context["percentile_label"] for context in contexts
    ]
    enriched["historical_sample_size"] = [
        context["sample_size"] for context in contexts
    ]
    enriched["historical_reference"] = [
        context["reference_description"] for context in contexts
    ]
    enriched["historical_reference_level"] = [
        context["reference_level"] for context in contexts
    ]
    enriched["historical_context_status"] = [
        context["status"] for context in contexts
    ]
    enriched["historical_context_reason"] = [
        context["reason"] for context in contexts
    ]
    enriched["historical_interpretation"] = [
        context["interpretation"] for context in contexts
    ]
    return enriched
