# GridLoad Intelligence BR

Plataforma de ciência de dados para análise e previsão da carga elétrica do
subsistema Nordeste, combinando dados públicos da ONS e do INMET.

## Executar o pipeline completo

A partir da raiz do projeto, execute:

```bash
python -m src.pipeline
```

O comando cria ou reutiliza `.venv`, instala as dependências de
`requirements.txt`, executa as ingestões da ONS e do INMET e, por fim, gera a
base consolidada em `data/processed/base.parquet`.

## Base processada

O arquivo `data/processed/base.parquet` é a base consolidada utilizada pelas
próximas etapas do projeto. Possui 40.896 registros horários, 22 variáveis e
cobre o período de 01/01/2022 a 31/08/2026. Cada linha representa uma hora do
subsistema Nordeste, combinando carga elétrica da ONS com as médias regionais
das estações meteorológicas do INMET.

| Variável | Tipo | Unidade | Descrição |
|---|---|---|---|
| `subsystem_code` | texto | — | Código do subsistema elétrico; nesta base, `NE`. |
| `timestamp` | datetime | — | Data e hora da observação, sem timezone após a consolidação. |
| `load_mw` | decimal | MWmed | Carga de energia horária do subsistema Nordeste. |
| `source_year` | inteiro | ano | Ano do arquivo anual da ONS que originou o registro. |
| `precipitation_mm` | decimal | mm | Precipitação média registrada pelas estações disponíveis. |
| `pressure_station_hpa` | decimal | hPa | Pressão atmosférica média no nível das estações. |
| `pressure_max_hpa` | decimal | hPa | Média regional da pressão máxima horária. |
| `pressure_min_hpa` | decimal | hPa | Média regional da pressão mínima horária. |
| `global_radiation_kj_m2` | decimal | kJ/m² | Radiação solar global média; ausências pontuais foram interpoladas linearmente. |
| `temperature_c` | decimal | °C | Temperatura média do ar. |
| `dew_point_c` | decimal | °C | Temperatura média do ponto de orvalho. |
| `temperature_max_c` | decimal | °C | Média regional da temperatura máxima horária. |
| `temperature_min_c` | decimal | °C | Média regional da temperatura mínima horária. |
| `dew_point_max_c` | decimal | °C | Média regional da temperatura máxima do ponto de orvalho. |
| `dew_point_min_c` | decimal | °C | Média regional da temperatura mínima do ponto de orvalho. |
| `humidity_max_pct` | decimal | % | Média regional da umidade relativa máxima horária. |
| `humidity_min_pct` | decimal | % | Média regional da umidade relativa mínima horária. |
| `humidity_pct` | decimal | % | Umidade relativa média do ar. |
| `wind_gust_ms` | decimal | m/s | Velocidade média das rajadas máximas de vento. |
| `wind_speed_ms` | decimal | m/s | Velocidade média horária do vento. |
| `wind_direction_deg` | decimal | graus | Direção regional do vento, calculada por média circular. |
| `station_count` | inteiro | estações | Quantidade de estações que contribuíram no horário. |

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
