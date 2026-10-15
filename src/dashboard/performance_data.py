"""Dados de avaliacao historica exibidos pela aplicacao."""

from __future__ import annotations

import pandas as pd

from src.dashboard.config import HORIZONS, TEST_METRICS


def test_metrics_frame() -> pd.DataFrame:
    rows = [
        {"horizon": horizon, **TEST_METRICS[horizon]}
        for horizon in HORIZONS
    ]
    return pd.DataFrame(rows)
