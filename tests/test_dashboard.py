from __future__ import annotations

import numpy as np
import pandas as pd

from src.dashboard.charts import recent_load_chart
from src.dashboard.config import TEST_METRICS
from src.dashboard.explainability import group_shap_values
from src.dashboard.forecast_service import forecast_summary
from src.dashboard.formatting import format_mw, format_percent
from src.dashboard.performance_data import test_metrics_frame as build_test_metrics_frame


def _snapshot() -> dict:
    origin = pd.Timestamp("2026-09-30 23:00:00")
    forecasts = {}
    for horizon, hours, value, percentile in (
        ("1h", 1, 1100.0, 75.0),
        ("6h", 6, 900.0, 40.0),
        ("24h", 24, 1200.0, 90.0),
    ):
        forecasts[horizon] = {
            "horizon": horizon,
            "forecast_mw": value,
            "target_timestamp": origin + pd.Timedelta(hours=hours),
            "delta_mw": value - 1000.0,
            "delta_pct": (value - 1000.0) / 10,
            "percentile": percentile,
            "percentile_label": f"P{int(percentile)}",
            "historical_context_status": "available",
        }
    return {"current_load_mw": 1000.0, "forecasts": forecasts}


def test_cyclical_shap_components_are_grouped_by_concept() -> None:
    grouped = group_shap_values(
        np.array([[10.0, -3.0, 4.0, 2.0, 1.5]]),
        ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "load_mw"],
    )

    assert grouped.loc[0, "Horário do dia"] == 7.0
    assert grouped.loc[0, "Dia da semana"] == 6.0
    assert grouped.loc[0, "Carga atual"] == 1.5


def test_forecast_markers_are_not_connected_or_interpolated() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-09-30 20:00", periods=4, freq="h"),
            "load_mw": [950.0, 980.0, 990.0, 1000.0],
        }
    )
    figure = recent_load_chart(history, _snapshot())

    assert figure.data[0].mode == "lines"
    assert all(trace.mode == "markers" for trace in figure.data[1:])
    assert all(len(trace.x) == 1 for trace in figure.data[1:])


def test_historical_test_metrics_match_published_values() -> None:
    frame = build_test_metrics_frame().set_index("horizon")

    assert frame.loc["1h"].to_dict() == TEST_METRICS["1h"]
    assert frame.loc["6h"].to_dict() == TEST_METRICS["6h"]
    assert frame.loc["24h"].to_dict() == TEST_METRICS["24h"]


def test_portuguese_number_formatting() -> None:
    assert format_mw(14963.25, 2) == "14.963,25 MW"
    assert format_percent(-3.06, 2, signed=True) == "-3,06%"


def test_summary_is_rule_based_and_uses_current_values() -> None:
    summary = forecast_summary(_snapshot())

    assert "1.000 MW" in summary
    assert "24h" in summary
    assert "P90" in summary
