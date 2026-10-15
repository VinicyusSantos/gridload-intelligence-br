"""Gera o artefato SHAP global consumido pelo dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.config import GLOBAL_SHAP_PATH  # noqa: E402
from src.dashboard.explainability import build_global_shap_artifact  # noqa: E402


def main() -> None:
    artifact = build_global_shap_artifact()
    print(f"Registros gerados: {len(artifact):,}")
    print(f"Artefato salvo: {GLOBAL_SHAP_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
