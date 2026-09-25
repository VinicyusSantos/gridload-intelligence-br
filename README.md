# GridLoad Intelligence BR

Plataforma de ciência de dados para análise e previsão da carga elétrica do
subsistema Nordeste, combinando dados públicos da ONS e do INMET.

## Inventário dos dados intermediários

Inventário gerado a partir dos arquivos disponíveis em `data/interim/`:

| Arquivo | Conteúdo e granularidade | Período disponível | Registros | Colunas | Tamanho | Chave |
|---|---|---:|---:|---:|---:|---|
| `load_clean.parquet` | Carga elétrica horária do subsistema Nordeste (ONS) | 01/01/2022 a 23/09/2026 | 41.448 | 5 | 0,67 MB | `subsystem_code`, `timestamp` |
| `weather_clean.parquet` | Média meteorológica horária regional do Nordeste (INMET) | 01/01/2022 a 31/08/2026 | 40.896 | 20 | 5,57 MB | `timestamp` |
| `weather_stations_clean.parquet` | Medições horárias detalhadas por estação automática (INMET) | 01/01/2022 a 31/08/2026 | 5.771.832 | 27 | 70,75 MB | `station_code`, `timestamp` |

### ONS — carga elétrica

O arquivo `load_clean.parquet` contém apenas o subsistema Nordeste, identificado
por `NE`, sem valores ausentes ou chaves duplicadas. As principais variáveis são:

- `timestamp`: data e hora da observação;
- `subsystem_code` e `subsystem`: identificação do subsistema;
- `load_mw`: carga de energia em MWmed;
- `source_year`: ano do arquivo de origem.

### INMET — meteorologia regional

O Instituto Nacional de Meteorologia (INMET) é a fonte dos dados meteorológicos
do projeto. Foram utilizados os arquivos históricos anuais das estações
automáticas localizadas nos nove estados do Nordeste, cobrindo o período de 2022
a agosto de 2026. As observações possuem frequência horária e horário informado
em UTC.

As variáveis disponíveis incluem:

| Grupo | Variáveis | Unidade |
|---|---|---|
| Precipitação | `precipitation_mm` | mm |
| Pressão atmosférica | `pressure_station_hpa`, `pressure_min_hpa`, `pressure_max_hpa` | hPa |
| Radiação solar | `global_radiation_kj_m2` | kJ/m² |
| Temperatura do ar | `temperature_c`, `temperature_min_c`, `temperature_max_c` | °C |
| Ponto de orvalho | `dew_point_c`, `dew_point_min_c`, `dew_point_max_c` | °C |
| Umidade relativa | `humidity_pct`, `humidity_min_pct`, `humidity_max_pct` | % |
| Vento | `wind_speed_ms`, `wind_gust_ms`, `wind_direction_deg` | m/s e graus |

O arquivo `weather_clean.parquet` possui uma linha por hora, calculada a partir
das estações disponíveis no Nordeste. Inclui precipitação, pressão atmosférica,
radiação global, temperatura, ponto de orvalho, umidade e vento. A coluna
`station_count` informa quantas estações contribuíram em cada horário.

A direção do vento é consolidada com média circular. As demais variáveis são
agregadas por média aritmética. Não houve preenchimento artificial de valores
ausentes; apenas a radiação global apresenta ausência residual de aproximadamente
0,01% na base regional.

### INMET — dados por estação

O arquivo `weather_stations_clean.parquet` preserva as medições individuais de
149 estações automáticas distribuídas pelos nove estados do Nordeste: AL, BA,
CE, MA, PB, PE, PI, RN e SE. Além das variáveis meteorológicas, contém código,
nome, UF, latitude, longitude, altitude e arquivo de origem de cada estação.

Essa base apresenta valores ausentes em diferentes proporções, causados por
indisponibilidade de estações e sensores. Os valores não foram imputados nesta
etapa. Não existem duplicações para a chave `station_code` + `timestamp`.

> **Atenção temporal:** os timestamps do INMET estão em UTC. O arquivo da ONS
> ainda não possui timezone explícito. Antes de integrar as duas fontes, é
> necessário confirmar e padronizar a referência temporal da ONS.
