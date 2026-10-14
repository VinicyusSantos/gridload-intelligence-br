from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    engineer_features,
    engineer_inference_features,
)
from src.inference.generate_forecasts import generate_forecasts
from src.models.train_models import ModelSpec
from src.pipeline import PIPELINE_STEPS


class FakeModel:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if list(frame.columns) != ["load_mw"]:
            raise AssertionError("A ordem das features foi alterada")
        return np.array([self.prediction])


class InferenceTest(unittest.TestCase):
    def test_inference_is_the_last_pipeline_step(self) -> None:
        step_name, script = PIPELINE_STEPS[-1]

        self.assertEqual(step_name, "Geracao das previsoes mais recentes")
        self.assertEqual(script.name, "generate_forecasts.py")

    def test_inference_features_preserve_latest_observation(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=240, freq="h"),
                "load_mw": np.linspace(10_000, 11_000, 240),
            }
        )

        inference = engineer_inference_features(frame)
        training = engineer_features(frame)

        self.assertEqual(inference["timestamp"].max(), frame["timestamp"].max())
        self.assertEqual(
            training["timestamp"].max(),
            frame["timestamp"].max() - pd.Timedelta(hours=24),
        )

    def test_each_horizon_is_added_to_the_forecast_origin(self) -> None:
        origin = pd.Timestamp("2026-09-30 23:00:00")
        frame = pd.DataFrame(
            {
                "timestamp": [origin],
                "load_mw": [12_000.0],
            }
        )
        specs = tuple(
            ModelSpec(
                horizon=horizon,
                features=("load_mw",),
                target=f"target_load_{horizon}",
                params={},
                output_path=Path(f"xgb_{horizon}.json"),
            )
            for horizon in ("1h", "6h", "24h")
        )
        predictions = {
            "xgb_1h.json": 12_100.0,
            "xgb_6h.json": 12_200.0,
            "xgb_24h.json": 12_300.0,
        }

        forecasts = generate_forecasts(
            frame,
            specs,
            {"trained_at_utc": "2026-10-14T00:00:00+00:00"},
            model_loader=lambda path: FakeModel(predictions[path.name]),
            generated_at=datetime(2026, 10, 14, tzinfo=timezone.utc),
        )

        self.assertEqual(
            forecasts["forecast_timestamp"].tolist(),
            [
                origin + pd.Timedelta(hours=1),
                origin + pd.Timedelta(hours=6),
                origin + pd.Timedelta(hours=24),
            ],
        )
        self.assertEqual(
            forecasts["predicted_load_mw"].tolist(),
            [12_100.0, 12_200.0, 12_300.0],
        )


if __name__ == "__main__":
    unittest.main()
