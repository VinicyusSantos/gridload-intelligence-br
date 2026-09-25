import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


CATALOG_URL = (
    "https://dados.ons.org.br/api/3/action/"
    "package_show?id=curva-carga"
)

YEARS = range(2022, 2027)

snapshot = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
raw_dir = Path("data/raw/ons/curva_carga") / snapshot
output_path = Path("data/interim/load_clean.parquet")

raw_dir.mkdir(parents=True, exist_ok=True)
output_path.parent.mkdir(parents=True, exist_ok=True)


def open_url(url: str):
    request = Request(
        url,
        headers={"User-Agent": "gridload-intelligence-br/0.1"},
    )
    return urlopen(request, timeout=120)


# Consulta o catálogo, evitando URLs fixas.
with open_url(CATALOG_URL) as response:
    catalog = json.load(response)["result"]

(raw_dir / "catalog.json").write_text(
    json.dumps(catalog, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

resources = {}

for resource in catalog["resources"]:
    match = re.fullmatch(
        r"CurvaCarga-(\d{4})",
        resource.get("name", ""),
    )

    if match and resource.get("format", "").upper() == "CSV":
        resources[int(match.group(1))] = resource["url"]

frames = []

for year in YEARS:
    if year not in resources:
        raise RuntimeError(f"CSV de {year} não encontrado no catálogo")

    destination = raw_dir / f"CURVA_CARGA_{year}.csv"

    with open_url(resources[year]) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)

    frame = pd.read_csv(
        destination,
        sep=";",
        encoding="utf-8",
        dtype={"id_subsistema": "string"},
    )

    frame["source_year"] = year
    frames.append(frame)

load = pd.concat(frames, ignore_index=True)

load = load.rename(
    columns={
        "id_subsistema": "subsystem_code",
        "nom_subsistema": "subsystem",
        "din_instante": "timestamp",
        "val_cargaenergiahomwmed": "load_mw",
    }
)

load["timestamp"] = pd.to_datetime(
    load["timestamp"],
    format="%Y-%m-%d %H:%M:%S",
    errors="raise",
)

load["load_mw"] = pd.to_numeric(
    load["load_mw"],
    errors="raise",
)

# MVP: apenas Nordeste.
load = load.loc[load["subsystem_code"] == "NE"].copy()
load = load.sort_values("timestamp").reset_index(drop=True)

if load[["timestamp", "load_mw"]].isna().any().any():
    raise ValueError("Foram encontrados valores ausentes")

duplicates = load.duplicated(["subsystem_code", "timestamp"])

if duplicates.any():
    raise ValueError(
        f"Foram encontrados {duplicates.sum()} timestamps duplicados"
    )

if (load["load_mw"] < 0).any():
    raise ValueError("Foram encontrados valores negativos de carga")

load.to_parquet(output_path, index=False)

print(f"Registros: {len(load):,}")
print(f"Período: {load['timestamp'].min()} até {load['timestamp'].max()}")
print(f"Arquivo criado: {output_path}")