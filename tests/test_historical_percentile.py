from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.context.historical_percentile import (
    add_historical_context_to_forecasts,
    historical_percentile,
    historical_percentiles_for_forecasts,
)


def weekly_history(periods: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-02 18:00:00",
                periods=periods,
                freq="7D",
            ),
            "load_mw": np.arange(1, periods + 1, dtype=float),
        }
    )


class HistoricalPercentileTest(unittest.TestCase):
    def test_uses_target_weekday_and_hour(self) -> None:
        history = weekly_history()
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "timestamp": [pd.Timestamp("2025-12-01 17:00:00")],
                        "load_mw": [1_000_000.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        target = history.loc[:99, "timestamp"].max() + pd.Timedelta(days=7)

        result = historical_percentile(
            history,
            prediction=94.0,
            target_timestamp=target,
            min_samples=30,
        )

        self.assertEqual(result["percentile"], 94.0)
        self.assertEqual(result["percentile_label"], "P94")
        self.assertEqual(result["sample_size"], 100)
        self.assertEqual(result["reference_weekday"], 1)
        self.assertEqual(result["reference_hour"], 18)
        self.assertEqual(result["reference_description"], "terças-feiras às 18h")

    def test_excludes_target_and_future_observations_for_backtesting(self) -> None:
        history = weekly_history(50)
        target = history["timestamp"].max() + pd.Timedelta(days=7)
        future = pd.DataFrame(
            {
                "timestamp": [target, target + pd.Timedelta(days=7)],
                "load_mw": [0.0, 0.0],
            }
        )
        history = pd.concat([history, future], ignore_index=True)

        result = historical_percentile(
            history,
            prediction=25.0,
            target_timestamp=target,
            min_samples=1,
        )

        self.assertEqual(result["sample_size"], 50)
        self.assertEqual(result["percentile"], 50.0)

    def test_uses_holiday_group_when_it_has_enough_samples(self) -> None:
        history = weekly_history(40)
        history["is_holiday"] = [True] * 10 + [False] * 30
        target = history["timestamp"].max() + pd.Timedelta(days=7)

        result = historical_percentile(
            history,
            prediction=5.0,
            target_timestamp=target,
            min_samples=10,
            use_holiday=True,
            target_is_holiday=True,
        )

        self.assertEqual(result["sample_size"], 10)
        self.assertTrue(result["holiday_filter_used"])
        self.assertEqual(result["reference_level"], "weekday_hour_holiday")
        self.assertEqual(result["percentile"], 50.0)

    def test_falls_back_when_holiday_group_is_too_small(self) -> None:
        history = weekly_history(40)
        history["is_holiday"] = [True] * 10 + [False] * 30
        target = history["timestamp"].max() + pd.Timedelta(days=7)

        result = historical_percentile(
            history,
            prediction=20.0,
            target_timestamp=target,
            min_samples=15,
            use_holiday=True,
            target_is_holiday=True,
        )

        self.assertEqual(result["sample_size"], 40)
        self.assertFalse(result["holiday_filter_used"])
        self.assertEqual(result["reference_level"], "weekday_hour")
        self.assertEqual(result["fallback_reason"], "insufficient_holiday_samples")

    def test_returns_reason_when_sample_is_insufficient(self) -> None:
        history = weekly_history(5)
        target = history["timestamp"].max() + pd.Timedelta(days=7)

        result = historical_percentile(
            history,
            prediction=3.0,
            target_timestamp=target,
            min_samples=30,
        )

        self.assertIsNone(result["percentile"])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason_code"], "insufficient_samples")
        self.assertEqual(result["sample_size"], 5)

    def test_does_not_silently_remove_duplicate_timestamps(self) -> None:
        history = weekly_history(40)
        history = pd.concat([history, history.iloc[[0]]], ignore_index=True)
        target = history["timestamp"].max() + pd.Timedelta(days=7)

        result = historical_percentile(
            history,
            prediction=20.0,
            target_timestamp=target,
            min_samples=1,
        )

        self.assertIsNone(result["percentile"])
        self.assertEqual(
            result["reason_code"],
            "duplicate_historical_timestamps",
        )
        self.assertEqual(result["duplicate_timestamp_count"], 1)

    def test_handles_timezone_and_rejects_naive_aware_mismatch(self) -> None:
        history = weekly_history(40)
        history["timestamp"] = history["timestamp"].dt.tz_localize(
            "America/Recife"
        )
        target = history["timestamp"].max() + pd.Timedelta(days=7)

        available = historical_percentile(
            history,
            prediction=20.0,
            target_timestamp=target,
            min_samples=1,
        )
        mismatch = historical_percentile(
            history,
            prediction=20.0,
            target_timestamp=target.tz_localize(None),
            min_samples=1,
        )

        self.assertEqual(available["status"], "available")
        self.assertEqual(mismatch["reason_code"], "timezone_mismatch")

    def test_handles_empty_invalid_and_extreme_cases(self) -> None:
        empty = historical_percentile(
            pd.DataFrame(columns=["timestamp", "load_mw"]),
            prediction=10.0,
            target_timestamp="2026-01-01 00:00:00",
        )
        invalid_prediction = historical_percentile(
            weekly_history(40),
            prediction=np.nan,
            target_timestamp="2026-01-01 00:00:00",
        )
        invalid_timestamp = historical_percentile(
            weekly_history(40),
            prediction=10.0,
            target_timestamp="nao-e-uma-data",
        )
        history = weekly_history(40)
        target = history["timestamp"].max() + pd.Timedelta(days=7)
        percentile_zero = historical_percentile(
            history, 0.0, target, min_samples=1
        )
        percentile_hundred = historical_percentile(
            history, 100.0, target, min_samples=1
        )

        self.assertEqual(empty["reason_code"], "empty_history")
        self.assertEqual(invalid_prediction["reason_code"], "invalid_prediction")
        self.assertEqual(invalid_timestamp["reason_code"], "invalid_target_timestamp")
        self.assertEqual(percentile_zero["percentile_label"], "P0")
        self.assertEqual(percentile_hundred["percentile_label"], "P100")

    def test_calculates_all_horizons_and_enriches_forecast_frame(self) -> None:
        history = weekly_history(100)
        target = history["timestamp"].max() + pd.Timedelta(days=7)
        targets = {
            "1h": target,
            "6h": target,
            "24h": target,
        }
        forecasts = {"1h": 10.0, "6h": 50.0, "24h": 90.0}

        contexts = historical_percentiles_for_forecasts(
            history,
            forecasts,
            targets,
        )
        forecast_frame = pd.DataFrame(
            {
                "horizon": list(forecasts),
                "predicted_load_mw": list(forecasts.values()),
                "forecast_timestamp": list(targets.values()),
            }
        )
        enriched = add_historical_context_to_forecasts(
            history,
            forecast_frame,
        )

        self.assertEqual(set(contexts), {"1h", "6h", "24h"})
        self.assertEqual(contexts["6h"]["percentile_label"], "P50")
        self.assertEqual(
            enriched["historical_percentile_label"].tolist(),
            ["P10", "P50", "P90"],
        )


if __name__ == "__main__":
    unittest.main()
