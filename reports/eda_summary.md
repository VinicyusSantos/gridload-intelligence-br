# Resumo da Análise Exploratória de Dados

## 1. Objetivo

Este relatório consolida os principais resultados da análise exploratória realizada sobre a base horária do subsistema Nordeste (`NE`). A variável-alvo é `load_mw`, que representa a carga elétrica média observada em cada hora, em MWmed.

Os valores deste documento foram recalculados a partir do estado atual de `data/processed/base.parquet`. A análise reproduzida no notebook está em `notebooks/03_eda.ipynb`.

## 2. Resumo executivo

- A base possui **40.896 observações horárias**, entre **01/01/2022 00h** e **31/08/2026 23h**.
- Não foram encontrados valores ausentes, timestamps duplicados ou interrupções na frequência horária.
- A carga média foi de **12.557,30 MWmed**, com mínimo de **7.262,28 MWmed** e máximo de **18.156,97 MWmed**.
- Existe uma tendência de crescimento da carga ao longo dos anos. A média passou de **11.171,08 MWmed em 2022** para **13.266,86 MWmed em 2025**.
- O ciclo intradiário é forte: a menor carga média ocorre às **06h** e a maior às **21h**.
- A carga é menor nos fins de semana, principalmente no domingo.
- Novembro apresenta a maior carga média mensal; junho e julho apresentam as menores.
- Temperatura e carga possuem associação positiva moderada, com correlação de Pearson de **0,4475**.
- A carga possui forte dependência temporal: `t-1h` apresenta autocorrelação de **0,9695** e `t-168h`, correspondente à mesma hora da semana anterior, de **0,9069**.
- As rampas horárias têm média próxima de zero, mas podem chegar a variações extremas de aproximadamente **+3.596 MWmed** e **-3.255 MWmed**.
- O limite P99 da carga é **15.529,36 MWmed**. Foram identificadas 409 horas acima desse patamar.

## 3. Escopo e qualidade dos dados

| Item | Resultado |
|---|---:|
| Subsistema | Nordeste (`NE`) |
| Início | 01/01/2022 00h |
| Fim | 31/08/2026 23h |
| Dias cobertos | 1.704 |
| Frequência | Horária |
| Linhas | 40.896 |
| Colunas | 22 |
| Variáveis numéricas | 20 |
| Valores ausentes | 0 |
| Timestamps duplicados | 0 |
| Estações meteorológicas por hora | 135 a 144 |

A base combina a carga da ONS com variáveis meteorológicas agregadas do INMET. Os campos podem ser organizados em:

- Identificação e tempo: `subsystem_code`, `timestamp` e `source_year`.
- Alvo: `load_mw`.
- Precipitação e radiação: `precipitation_mm` e `global_radiation_kj_m2`.
- Pressão: `pressure_station_hpa`, `pressure_max_hpa` e `pressure_min_hpa`.
- Temperatura: `temperature_c`, `temperature_max_c` e `temperature_min_c`.
- Ponto de orvalho: `dew_point_c`, `dew_point_max_c` e `dew_point_min_c`.
- Umidade: `humidity_pct`, `humidity_max_pct` e `humidity_min_pct`.
- Vento: `wind_speed_ms`, `wind_gust_ms` e `wind_direction_deg`.
- Cobertura meteorológica: `station_count`.

Embora não existam nulos na base processada, isso não elimina a necessidade de avaliar a qualidade dos valores agregados e eventuais anomalias provenientes das fontes.

## 4. Distribuição da carga

| Estatística | `load_mw` |
|---|---:|
| Média | 12.557,30 MWmed |
| Desvio-padrão | 1.510,27 MWmed |
| Mínimo | 7.262,28 MWmed |
| P25 | 11.528,35 MWmed |
| Mediana | 12.603,67 MWmed |
| P75 | 13.665,22 MWmed |
| Máximo | 18.156,97 MWmed |

A média e a mediana são próximas, mas a distribuição possui caudas relevantes. Os valores extremos devem ser tratados com atenção porque podem representar eventos reais de demanda, mudanças operacionais ou problemas de qualidade.

## 5. Evolução temporal e tendência

### 5.1 Média anual

| Ano | Carga média | Observações |
|---|---:|---:|
| 2022 | 11.171,08 MWmed | 8.760 |
| 2023 | 12.116,96 MWmed | 8.760 |
| 2024 | 13.121,60 MWmed | 8.784 |
| 2025 | 13.266,86 MWmed | 8.760 |
| 2026 | 13.385,18 MWmed | 5.832 |

Entre 2022 e 2025, a carga média anual cresceu aproximadamente **18,8%**. O valor de 2026 não é diretamente comparável aos anos completos porque cobre apenas janeiro a agosto.

### 5.2 Decomposição sazonal

A série foi agregada por dia e decomposta com `seasonal_decompose`, usando:

- Modelo aditivo.
- Período de 365 dias.
- Componentes de série observada, tendência, sazonalidade anual e resíduo.

A tendência estimada passou de aproximadamente **10.517 MWmed** no início de 2022 para **13.234 MWmed** no final de agosto de 2026, uma diferença aproximada de **2.717 MWmed**. O maior valor da tendência estimada ocorreu em março de 2026, próximo de **13.402 MWmed**.

Os valores nas extremidades da tendência devem ser interpretados com cautela, pois o método utiliza extrapolação. A decomposição anual também não substitui as análises separadas dos ciclos horário e semanal.

## 6. Sazonalidade horária, semanal e mensal

### 6.1 Curva média por hora

- Menor carga média: **06h**, com **11.304,24 MWmed**.
- Maior carga média: **21h**, com **13.806,48 MWmed**.
- Diferença entre o vale e o pico médio: aproximadamente **2.502 MWmed**.

A carga diminui durante a madrugada, atinge o vale no início da manhã e cresce ao longo do dia. O período entre 18h e 23h concentra os maiores valores médios.

### 6.2 Curva média por dia da semana

| Dia | Carga média |
|---|---:|
| Segunda | 12.681,62 MWmed |
| Terça | 12.878,01 MWmed |
| Quarta | 12.901,14 MWmed |
| Quinta | 12.903,57 MWmed |
| Sexta | 12.839,13 MWmed |
| Sábado | 12.234,84 MWmed |
| Domingo | 11.468,11 MWmed |

Quinta-feira possui a maior média, embora terça, quarta e sexta apresentem valores muito próximos. A redução no sábado e principalmente no domingo evidencia o ciclo semanal da atividade econômica e social.

### 6.3 Curva média mensal

| Mês | Carga média |
|---|---:|
| Janeiro | 12.646,06 MWmed |
| Fevereiro | 12.811,34 MWmed |
| Março | 12.770,09 MWmed |
| Abril | 12.579,04 MWmed |
| Maio | 12.448,40 MWmed |
| Junho | 11.994,89 MWmed |
| Julho | 11.947,04 MWmed |
| Agosto | 12.333,77 MWmed |
| Setembro | 12.363,62 MWmed |
| Outubro | 12.935,15 MWmed |
| Novembro | 13.154,86 MWmed |
| Dezembro | 12.955,24 MWmed |

Novembro apresenta a maior carga média e julho a menor. Existe uma recuperação clara da carga entre setembro e novembro.

### 6.4 Heatmap de dia da semana e hora

- Maior combinação média: **quinta-feira às 21h**, com **14.048,10 MWmed**.
- Menor combinação média: **domingo às 09h**, com **9.907,52 MWmed**.

O heatmap confirma que o pico noturno dos dias úteis e a redução do fim de semana são padrões consistentes e combinados.

## 7. Temperatura e carga

A temperatura média agregada variou entre **18,23 °C** e **35,36 °C**. A correlação de Pearson entre `temperature_c` e `load_mw` foi de **0,4475**, indicando associação positiva moderada.

A curva por faixas de 1 °C reforça o padrão geral:

- Em torno de 18,8 °C, a carga média foi aproximadamente 10.829 MWmed.
- Entre 24 °C e 26 °C, a carga média ficou entre aproximadamente 12.652 e 13.000 MWmed.
- Entre 32 °C e 34 °C, a carga média ficou entre aproximadamente 13.828 e 14.002 MWmed.

A faixa acima de 35 °C possui apenas oito observações e não deve sustentar conclusões isoladamente. A relação temperatura-carga também pode ser confundida por hora, mês, ano e comportamento econômico; correlação não implica causalidade.

## 8. Correlações entre variáveis

### 8.1 Maiores correlações com `load_mw`

| Variável | Correlação de Pearson |
|---|---:|
| `source_year` | 0,5261 |
| `temperature_min_c` | 0,4921 |
| `temperature_max_c` | 0,4692 |
| `temperature_c` | 0,4475 |
| `station_count` | -0,4230 |
| `humidity_max_pct` | -0,3849 |
| `humidity_min_pct` | -0,3808 |
| `humidity_pct` | -0,3529 |
| `wind_gust_ms` | 0,3322 |
| `wind_speed_ms` | 0,3074 |

`source_year` captura principalmente a tendência temporal de crescimento, não uma relação causal direta. A correlação negativa de `station_count` merece atenção: essa variável representa cobertura dos dados meteorológicos e pode funcionar como marcador indireto do período ou do processo de coleta.

Precipitação e ponto de orvalho apresentaram correlações lineares fracas com a carga quando avaliados isoladamente. Isso não exclui relações não lineares ou interações com outras variáveis.

### 8.2 Multicolinearidade meteorológica

Foram observadas correlações muito altas entre medições da mesma família:

| Par de variáveis | Correlação |
|---|---:|
| `pressure_min_hpa` × `pressure_max_hpa` | 0,9967 |
| `temperature_max_c` × `temperature_c` | 0,9948 |
| `humidity_pct` × `humidity_min_pct` | 0,9946 |
| `pressure_min_hpa` × `pressure_station_hpa` | 0,9940 |
| `dew_point_min_c` × `dew_point_c` | 0,9939 |
| `temperature_min_c` × `temperature_max_c` | 0,9910 |
| `wind_speed_ms` × `wind_gust_ms` | 0,9806 |

Essa redundância pode prejudicar a interpretação de modelos lineares e aumentar a instabilidade dos coeficientes. Para esses modelos, convém selecionar uma variável representativa por grupo, aplicar regularização ou criar amplitudes, como temperatura máxima menos mínima.

## 9. Autocorrelação da carga

| Defasagem | Autocorrelação |
|---|---:|
| `t-1h` | 0,9695 |
| `t-2h` | 0,9081 |
| `t-24h` | 0,8183 |
| `t-48h` | 0,6958 |
| `t-168h` | 0,9069 |

Principais conclusões:

- A carga da hora anterior é o preditor temporal isolado mais forte entre os atrasos testados.
- A autocorrelação de `t-24h` confirma a repetição do ciclo diário.
- O valor elevado de `t-168h` confirma um padrão semanal forte: a mesma hora da semana anterior tende a ser muito informativa.
- A redução em `t-48h` mostra que a semelhança não depende apenas da distância temporal; o alinhamento com o ciclo semanal é importante.

Esses resultados sustentam a criação de variáveis de lag, mas exigem divisão temporal dos dados para evitar vazamento de informação.

## 10. Rampas horárias

A rampa foi definida como:

\[
Ramp_t = L_t - L_{t-1}
\]

Valores positivos representam aumento de carga; valores negativos representam redução.

### 10.1 Estatísticas gerais e cauda positiva

| Estatística | Rampa |
|---|---:|
| Média | 0,11 MWmed |
| Desvio-padrão | 373,17 MWmed |
| P90 | 489,00 MWmed |
| P95 | 655,42 MWmed |
| P97 | 795,92 MWmed |
| P99 | 1.085,02 MWmed |
| Máximo | 3.595,71 MWmed |

A média próxima de zero indica que aumentos e reduções se compensam no período completo. Entretanto, o desvio-padrão e as caudas demonstram variações horárias relevantes.

O P99 indica que apenas 1% de todas as transições horárias possui aumento superior a aproximadamente 1.085 MWmed.

### 10.2 Cauda negativa

| Estatística | Rampa |
|---|---:|
| P10 | -418,90 MWmed |
| P5 | -507,85 MWmed |
| P3 | -568,67 MWmed |
| P1 | -710,87 MWmed |
| Mínimo | -3.255,47 MWmed |

O P1 indica que apenas 1% das transições apresenta uma queda superior a aproximadamente 711 MWmed em magnitude. A comparação entre P99 e P1 sugere que a cauda positiva é mais intensa que a negativa nos percentis analisados.

## 11. Picos de carga

| Percentil | Limite | Horas acima | Proporção |
|---|---:|---:|---:|
| P90 | 14.531,18 MWmed | 4.090 | 10% |
| P95 | 14.929,50 MWmed | 2.045 | 5% |
| P97 | 15.150,98 MWmed | 1.227 | 3% |
| P99 | 15.529,36 MWmed | 409 | 1% |

Uma classificação operacional possível é:

- Normal: até P90.
- Elevada: entre P90 e P95.
- Muito elevada: entre P95 e P97.
- Crítica: entre P97 e P99.
- Extrema: acima de P99.

Essas categorias são relativas ao histórico analisado e não correspondem automaticamente a limites técnicos do sistema elétrico.

### 11.1 Dez maiores cargas observadas

| Posição | Timestamp | Carga |
|---:|---|---:|
| 1 | 08/02/2024 10h | 18.156,97 MWmed |
| 2 | 04/02/2026 22h | 16.987,37 MWmed |
| 3 | 04/02/2026 21h | 16.823,36 MWmed |
| 4 | 03/02/2026 22h | 16.619,25 MWmed |
| 5 | 04/02/2026 23h | 16.594,84 MWmed |
| 6 | 10/11/2024 00h | 16.468,33 MWmed |
| 7 | 03/02/2026 21h | 16.461,94 MWmed |
| 8 | 18/11/2025 22h | 16.446,62 MWmed |
| 9 | 06/05/2026 18h | 16.426,14 MWmed |
| 10 | 06/11/2025 23h | 16.415,69 MWmed |

O máximo de 08/02/2024 está 2.627,61 MWmed acima do P99 e 1.169,60 MWmed acima do segundo maior registro. Esse ponto deve ser verificado na fonte e comparado às horas vizinhas e às condições meteorológicas antes de ser tratado como anomalia ou removido.

