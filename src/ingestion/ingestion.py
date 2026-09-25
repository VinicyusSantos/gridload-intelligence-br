"""Executa todos os pipelines de ingestao do GridLoad Intelligence BR."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTION_DIR = Path(__file__).resolve().parent

PIPELINES = (
    ("ONS", INGESTION_DIR / "ingest_ons.py"),
    ("INMET", INGESTION_DIR / "ingest_inmet.py"),
)


def run_pipeline(name: str, script: Path) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"Iniciando ingestao: {name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f"Ingestao {name} concluida com sucesso.", flush=True)


def main() -> None:
    for name, script in PIPELINES:
        if not script.is_file():
            raise FileNotFoundError(f"Pipeline nao encontrado: {script}")
        run_pipeline(name, script)

    print(f"\n{'=' * 60}", flush=True)
    print("Todas as ingestoes foram concluidas com sucesso.", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"\nA ingestao falhou com codigo de saida {error.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(error.returncode) from error
