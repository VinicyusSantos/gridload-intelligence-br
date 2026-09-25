"""Ingestao dos dados meteorologicos historicos do INMET.

Baixa os ZIPs anuais originais para ``data/raw`` e produz em
``data/interim`` uma base limpa por estacao e uma base horaria regional.

Para limitar o processamento a estacoes especificas, defina a variavel
``INMET_STATION_CODES`` com codigos separados por virgula, por exemplo:
``INMET_STATION_CODES=A301,A305``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


YEARS = range(2022, 2027)
BASE_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
NORTHEAST_UFS = {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}

SNAPSHOT = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RAW_ROOT = Path("data/raw/inmet/historico_anual")
RAW_DIR = RAW_ROOT / SNAPSHOT
INTERIM_DIR = Path("data/interim")
STATIONS_OUTPUT = INTERIM_DIR / "weather_stations_clean.parquet"
REGIONAL_OUTPUT = INTERIM_DIR / "weather_clean.parquet"

USER_AGENT = "gridload-intelligence-br/0.1"
DOWNLOAD_RETRIES = 3
MISSING_VALUES = ["", "null", "NULL", "NaN", "-9999", "-9999,0", "-9999.0"]


def normalize_label(value: str) -> str:
    """Remove acentos e pontuacao para comparar rotulos do INMET."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^A-Za-z0-9]+", " ", ascii_value)
    return re.sub(r"\s+", " ", ascii_value).strip().upper()


def parse_station_codes() -> set[str] | None:
    configured = os.getenv("INMET_STATION_CODES", "").strip()
    if not configured:
        return None
    codes = {code.strip().upper() for code in configured.split(",") if code.strip()}
    return codes or None


def open_url(url: str, timeout: int = 180):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(request, timeout=timeout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Baixa um arquivo de forma atomica e com novas tentativas."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with open_url(url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size == 0:
                raise RuntimeError("O INMET retornou um arquivo vazio")
            temporary.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2**attempt)

    raise RuntimeError(
        f"Falha ao baixar {url} apos {DOWNLOAD_RETRIES} tentativas. "
        f"Baixe-o manualmente para {destination}, se necessario."
    ) from last_error


def decode_csv(content: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Codificacao do CSV nao reconhecida")


def find_header_index(lines: list[str]) -> int:
    for index, line in enumerate(lines[:40]):
        fields = [normalize_label(field) for field in line.split(";")]
        if fields and fields[0].startswith("DATA") and any(
            "HORA" in field for field in fields
        ):
            return index
    raise ValueError("Cabecalho horario nao encontrado")


def parse_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        # Os arquivos anuais usam tanto ``UF: PE;`` quanto ``UF:;PE``.
        metadata[normalize_label(key)] = value.strip().strip(";").strip()
    return metadata


def latest_valid_archive(year: int) -> Path | None:
    """Localiza um ZIP ja baixado para evitar transferencias repetidas."""
    candidates = sorted(
        RAW_ROOT.glob(f"*/INMET_{year}.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.stat().st_size > 0 and zipfile.is_zipfile(candidate):
            return candidate
    return None


def metadata_value(metadata: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(normalize_label(key))
        if value is not None:
            return value
    return None


def parse_decimal(value: str | None) -> float:
    if value is None or not str(value).strip():
        return float("nan")
    return float(str(value).strip().replace(",", "."))


def filename_station_code(filename: str) -> str | None:
    match = re.search(r"(?:^|_)([A-Z]\d{3})(?:_|\.)", filename.upper())
    return match.group(1) if match else None


def filename_uf(filename: str) -> str | None:
    match = re.search(
        rf"(?:^|_)({'|'.join(sorted(NORTHEAST_UFS))})(?:_|\.)",
        filename.upper(),
    )
    return match.group(1) if match else None


def find_column(columns: list[str], required_tokens: tuple[str, ...]) -> str | None:
    for column in columns:
        normalized = normalize_label(column)
        if all(token in normalized for token in required_tokens):
            return column
    return None


def canonical_measurement_name(column: str) -> str:
    """Converte os nomes extensos do INMET para nomes estaveis."""
    label = normalize_label(column)
    rules = (
        (("PRECIPITACAO", "TOTAL"), "precipitation_mm"),
        (("PRESSAO", "ATMOSFERICA", "MAX"), "pressure_max_hpa"),
        (("PRESSAO", "ATMOSFERICA", "MIN"), "pressure_min_hpa"),
        (("PRESSAO", "ATMOSFERICA"), "pressure_station_hpa"),
        (("RADIACAO", "GLOBAL"), "global_radiation_kj_m2"),
        (("TEMPERATURA", "ORVALHO", "MAX"), "dew_point_max_c"),
        (("TEMPERATURA", "ORVALHO", "MIN"), "dew_point_min_c"),
        (("TEMPERATURA", "PONTO", "ORVALHO"), "dew_point_c"),
        (("TEMPERATURA", "MAX"), "temperature_max_c"),
        (("TEMPERATURA", "MIN"), "temperature_min_c"),
        (("TEMPERATURA", "BULBO", "SECO"), "temperature_c"),
        (("UMIDADE", "MAX"), "humidity_max_pct"),
        (("UMIDADE", "MIN"), "humidity_min_pct"),
        (("UMIDADE", "RELATIVA"), "humidity_pct"),
        (("VENTO", "DIRECAO"), "wind_direction_deg"),
        (("VENTO", "RAJADA"), "wind_gust_ms"),
        (("VENTO", "VELOCIDADE"), "wind_speed_ms"),
    )
    for tokens, canonical_name in rules:
        if all(token in label for token in tokens):
            return canonical_name
    return re.sub(r"_+", "_", label.lower().replace(" ", "_")).strip("_")


def parse_timestamp(frame: pd.DataFrame, date_column: str, hour_column: str) -> pd.Series:
    date_text = (
        frame[date_column]
        .astype("string")
        .str.strip()
        .str.replace("/", "-", regex=False)
    )
    # Os arquivos anuais usam YYYY/MM/DD. O formato explicito evita que
    # 2022/01/02 seja interpretado como 2 de janeiro em algumas versoes
    # do pandas e como 1 de fevereiro em outras.
    dates = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    unresolved = dates.isna() & date_text.notna()
    if unresolved.any():
        dates.loc[unresolved] = pd.to_datetime(
            date_text.loc[unresolved],
            errors="coerce",
            dayfirst=True,
        )
    hours_text = frame[hour_column].astype("string").str.strip()
    hhmm = hours_text.str.extract(r"(\d{1,4})", expand=False).str.zfill(4)
    hours = pd.to_numeric(hhmm.str[:2], errors="coerce")
    minutes = pd.to_numeric(hhmm.str[2:4], errors="coerce")
    valid_time = hours.between(0, 23) & minutes.between(0, 59)
    timestamp = dates.dt.normalize()
    timestamp = timestamp + pd.to_timedelta(hours.where(valid_time), unit="h")
    timestamp = timestamp + pd.to_timedelta(minutes.where(valid_time), unit="m")
    return timestamp.dt.tz_localize("UTC")


def parse_station_member(
    archive: zipfile.ZipFile,
    member_name: str,
    source_year: int,
    selected_codes: set[str] | None,
) -> pd.DataFrame | None:
    text = decode_csv(archive.read(member_name))
    lines = text.splitlines()
    header_index = find_header_index(lines)
    metadata = parse_metadata(lines[:header_index])

    station_code = (
        metadata_value(metadata, "CODIGO WMO", "CODIGO DA ESTACAO")
        or filename_station_code(member_name)
    )
    station_code = station_code.upper() if station_code else None
    uf = metadata_value(metadata, "UF") or filename_uf(member_name)
    uf = uf.upper() if uf else None

    if uf not in NORTHEAST_UFS:
        return None
    if selected_codes is not None and station_code not in selected_codes:
        return None

    frame = pd.read_csv(
        io.StringIO("\n".join(lines[header_index:])),
        sep=";",
        decimal=",",
        na_values=MISSING_VALUES,
        low_memory=False,
    ).dropna(axis="columns", how="all")

    columns = list(frame.columns)
    date_column = find_column(columns, ("DATA",))
    hour_column = find_column(columns, ("HORA", "UTC"))
    if date_column is None or hour_column is None:
        raise ValueError("Colunas de data/hora UTC nao encontradas")

    timestamp = parse_timestamp(frame, date_column, hour_column)
    measurement_columns = [
        column
        for column in columns
        if column not in {date_column, hour_column}
        and not normalize_label(column).startswith("UNNAMED")
    ]

    renamed: dict[str, str] = {}
    used_names: set[str] = set()
    for column in measurement_columns:
        name = canonical_measurement_name(column)
        if name in used_names:
            suffix = 2
            while f"{name}_{suffix}" in used_names:
                suffix += 1
            name = f"{name}_{suffix}"
        renamed[column] = name
        used_names.add(name)

    measurements = frame[measurement_columns].rename(columns=renamed).copy()
    for column in measurements.columns:
        measurements[column] = pd.to_numeric(measurements[column], errors="coerce")

    result = pd.DataFrame(
        {
            "timestamp": timestamp,
            "station_code": station_code,
            "station_name": metadata_value(metadata, "ESTACAO"),
            "uf": uf,
            "region": metadata_value(metadata, "REGIAO") or "NE",
            "latitude": parse_decimal(metadata_value(metadata, "LATITUDE")),
            "longitude": parse_decimal(metadata_value(metadata, "LONGITUDE")),
            "altitude_m": parse_decimal(metadata_value(metadata, "ALTITUDE")),
            "source_year": source_year,
            "source_file": member_name,
        }
    )
    return pd.concat(
        [result.reset_index(drop=True), measurements.reset_index(drop=True)],
        axis="columns",
    )


def validate_station_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.loc[frame["timestamp"].notna()].copy()
    frame = frame.sort_values(["station_code", "timestamp"]).reset_index(drop=True)

    duplicated = frame.duplicated(["station_code", "timestamp"], keep=False)
    if duplicated.any():
        raise ValueError(
            f"Foram encontrados {int(duplicated.sum())} registros duplicados "
            "por estacao e timestamp"
        )

    invalid_counts: dict[str, int] = {}
    valid_ranges = {
        "humidity_pct": (0, 100),
        "humidity_min_pct": (0, 100),
        "humidity_max_pct": (0, 100),
        "wind_direction_deg": (0, 360),
        "wind_speed_ms": (0, 150),
        "wind_gust_ms": (0, 150),
        "precipitation_mm": (0, 1000),
        "global_radiation_kj_m2": (0, 10000),
        "temperature_c": (-90, 65),
        "temperature_min_c": (-90, 65),
        "temperature_max_c": (-90, 65),
        "dew_point_c": (-100, 65),
        "dew_point_min_c": (-100, 65),
        "dew_point_max_c": (-100, 65),
        "pressure_station_hpa": (300, 1100),
        "pressure_min_hpa": (300, 1100),
        "pressure_max_hpa": (300, 1100),
    }
    for column, (minimum, maximum) in valid_ranges.items():
        if column not in frame.columns:
            continue
        invalid = frame[column].notna() & ~frame[column].between(minimum, maximum)
        invalid_counts[column] = int(invalid.sum())
        frame.loc[invalid, column] = np.nan
    return frame, invalid_counts


def circular_mean_degrees(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    radians = np.deg2rad(values.astype(float))
    angle = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    normalized_angle = float(angle % 360)
    return 0.0 if np.isclose(normalized_angle, 360.0) else normalized_angle


def aggregate_regional(frame: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = {
        "timestamp",
        "station_code",
        "station_name",
        "uf",
        "region",
        "latitude",
        "longitude",
        "altitude_m",
        "source_year",
        "source_file",
    }
    measurement_columns = [
        column
        for column in frame.columns
        if column not in metadata_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregation: dict[str, str | Callable[[pd.Series], float]] = {
        column: "mean"
        for column in measurement_columns
        if column != "wind_direction_deg"
    }
    if "wind_direction_deg" in measurement_columns:
        aggregation["wind_direction_deg"] = circular_mean_degrees

    regional = frame.groupby("timestamp", as_index=False).agg(aggregation)
    station_count = (
        frame.groupby("timestamp")["station_code"]
        .nunique()
        .rename("station_count")
        .reset_index()
    )
    regional = regional.merge(station_count, on="timestamp", how="left")
    regional.insert(1, "region", "NORDESTE")
    return regional.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    selected_codes = parse_station_codes()

    archives: list[tuple[int, Path]] = []
    manifest: dict[str, object] = {
        "source": "INMET - Dados Historicos Anuais",
        "snapshot_utc": SNAPSHOT,
        "selected_station_codes": sorted(selected_codes) if selected_codes else None,
        "archives": [],
        "skipped_files": [],
    }

    for year in YEARS:
        url = BASE_URL.format(year=year)
        destination = latest_valid_archive(year)
        if destination is None:
            destination = RAW_DIR / f"INMET_{year}.zip"
            print(f"Baixando dados do INMET de {year}...")
            download(url, destination)
        else:
            print(f"Reutilizando dados do INMET de {year}: {destination}")
        archives.append((year, destination))
        manifest["archives"].append(
            {
                "year": year,
                "url": url,
                "file": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    frames: list[pd.DataFrame] = []
    found_station_codes: set[str] = set()

    for year, archive_path in archives:
        print(f"Processando {archive_path.name}...")
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            ]
            for member_name in members:
                try:
                    frame = parse_station_member(
                        archive, member_name, year, selected_codes
                    )
                except (ValueError, UnicodeError, pd.errors.ParserError) as error:
                    manifest["skipped_files"].append(
                        {"year": year, "file": member_name, "reason": str(error)}
                    )
                    continue

                if frame is not None and not frame.empty:
                    frames.append(frame)
                    found_station_codes.update(frame["station_code"].dropna().unique())

    if not frames:
        raise RuntimeError("Nenhum dado de estacao do Nordeste foi encontrado")

    if selected_codes:
        missing_codes = selected_codes - found_station_codes
        if missing_codes:
            raise RuntimeError(
                "Estacoes configuradas nao encontradas: "
                + ", ".join(sorted(missing_codes))
            )

    stations = pd.concat(frames, ignore_index=True, sort=False)
    stations, invalid_counts = validate_station_data(stations)
    regional = aggregate_regional(stations)

    stations.to_parquet(STATIONS_OUTPUT, index=False)
    regional.to_parquet(REGIONAL_OUTPUT, index=False)

    manifest.update(
        {
            "station_rows": len(stations),
            "regional_rows": len(regional),
            "stations": sorted(found_station_codes),
            "period_start_utc": str(stations["timestamp"].min()),
            "period_end_utc": str(stations["timestamp"].max()),
            "invalid_values_replaced_with_null": invalid_counts,
            "station_output": str(STATIONS_OUTPUT),
            "regional_output": str(REGIONAL_OUTPUT),
        }
    )
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Estacoes utilizadas: {len(found_station_codes)}")
    print(f"Registros por estacao: {len(stations):,}")
    print(f"Registros regionais: {len(regional):,}")
    print(f"Periodo UTC: {stations['timestamp'].min()} ate {stations['timestamp'].max()}")
    print(f"Arquivo por estacao: {STATIONS_OUTPUT}")
    print(f"Arquivo regional: {REGIONAL_OUTPUT}")


if __name__ == "__main__":
    main()
