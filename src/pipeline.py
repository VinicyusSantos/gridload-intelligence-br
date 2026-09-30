"""Ponto unico de execucao do pipeline do GridLoad Intelligence BR.

Uso, a partir da raiz do projeto::

    python -m src.pipeline

O modulo cria ou reutiliza ``.venv``, instala as dependencias declaradas em
``requirements.txt`` e executa as etapas essenciais na ordem configurada em
``PIPELINE_STEPS``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

PIPELINE_STEPS = (
    ("Ingestao ONS + INMET", PROJECT_ROOT / "src/ingestion/ingestion.py"),
    ("Processamento ONS + INMET", PROJECT_ROOT / "src/processing/processing.py"),
    (
        "Engenharia de features",
        PROJECT_ROOT / "src/features/feature_engineering.py",
    ),
)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts/python.exe"
    return VENV_DIR / "bin/python"


def running_inside_project_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def ensure_venv() -> Path:
    python_path = venv_python()
    if not python_path.is_file():
        print(f"Criando ambiente virtual em {VENV_DIR.relative_to(PROJECT_ROOT)}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    if not python_path.is_file():
        raise FileNotFoundError(
            f"O Python do ambiente virtual nao foi encontrado: {python_path}"
        )
    return python_path


def restart_inside_venv(python_path: Path) -> None:
    """Executa novamente este modulo usando o Python do projeto."""
    print("Reiniciando o pipeline dentro do ambiente virtual...", flush=True)
    subprocess.run(
        [str(python_path), "-u", "-m", "src.pipeline"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def install_requirements() -> None:
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Arquivo de dependencias nao encontrado: {REQUIREMENTS}")

    print(f"\n{'=' * 70}", flush=True)
    print("Verificando dependencias do projeto", flush=True)
    print(f"{'=' * 70}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(REQUIREMENTS),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_step(name: str, script: Path) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"Etapa nao encontrada: {script}")

    print(f"\n{'=' * 70}", flush=True)
    print(f"Iniciando etapa: {name}", flush=True)
    print(f"{'=' * 70}", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"Etapa concluida: {name}", flush=True)


def run_project_pipeline() -> None:
    install_requirements()
    for name, script in PIPELINE_STEPS:
        run_step(name, script)

    print(f"\n{'=' * 70}", flush=True)
    print("Pipeline completo executado com sucesso.", flush=True)
    print(f"{'=' * 70}", flush=True)


def main() -> None:
    python_path = ensure_venv()
    if not running_inside_project_venv():
        restart_inside_venv(python_path)
        return
    run_project_pipeline()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline interrompido com codigo de saida {error.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(error.returncode) from error
