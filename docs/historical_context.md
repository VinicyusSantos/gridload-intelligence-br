# Contexto histórico das previsões

## Metodologia

Para cada previsão, a referência é formada somente pelas cargas observadas no
mesmo dia da semana e na mesma hora do `target_timestamp`. O horário usado é o
instante previsto, e não o instante em que a previsão foi gerada.

Para uma previsão `y_pred`, o percentil empírico é:

```text
percentile = média(histórico <= y_pred) * 100
```

O filtro também exige `timestamp < target_timestamp`. Assim, a mesma função é
segura para previsões futuras e para backtesting, sem leakage.

Por padrão, são exigidas pelo menos 30 observações. Quando o tratamento
opcional de feriado é ativado, tenta-se primeiro `weekday + hour + feriado`. Se
esse grupo não atingir o mínimo, a função volta automaticamente para
`weekday + hour`. Nenhum percentil é inventado quando nem o fallback possui
amostra suficiente.

## Uma previsão

```python
import pandas as pd

from src.context.historical_percentile import historical_percentile

history = pd.read_parquet("data/processed/base.parquet")

context = historical_percentile(
    history=history,
    prediction=14_920.0,
    target_timestamp="2026-10-20 18:00:00",
    min_samples=30,
)

print(context["percentile_label"])
print(context["interpretation"])
```

Exemplo de campos retornados:

```python
{
    "prediction_mw": 14920.0,
    "percentile": 94.2,
    "percentile_label": "P94",
    "sample_size": 198,
    "target_timestamp": "2026-10-20 18:00:00",
    "reference_weekday": 1,
    "reference_hour": 18,
    "reference_description": "terças-feiras às 18h",
    "status": "available",
}
```

## Previsões de 1h, 6h e 24h

```python
from src.context.historical_percentile import (
    historical_percentiles_for_forecasts,
)

contexts = historical_percentiles_for_forecasts(
    history=history,
    forecasts={
        "1h": forecast_1h,
        "6h": forecast_6h,
        "24h": forecast_24h,
    },
    target_timestamps={
        "1h": target_timestamp_1h,
        "6h": target_timestamp_6h,
        "24h": target_timestamp_24h,
    },
    min_samples=30,
)
```

Cada chave recebe a carga prevista, o percentil, o rótulo, o tamanho da
amostra e a descrição da referência utilizada.

## Feriados opcionais

Quando o histórico possui `is_holiday`, informe também o status do instante
previsto:

```python
context = historical_percentile(
    history=history,
    prediction=14_920.0,
    target_timestamp="2026-10-20 18:00:00",
    min_samples=30,
    use_holiday=True,
    holiday_col="is_holiday",
    target_is_holiday=False,
)
```

## Integração com a inferência

O adaptador de DataFrame recebe diretamente a saída dos três modelos:

```python
from src.context.historical_percentile import (
    add_historical_context_to_forecasts,
)

forecasts = add_historical_context_to_forecasts(
    history=history,
    forecasts=forecasts,
    prediction_col="predicted_load_mw",
    target_col="forecast_timestamp",
    min_samples=30,
)
```

Essa chamada já está posicionada depois de `generate_forecasts()` em
`src/inference/generate_forecasts.py`. Ela apenas enriquece a saída; não altera
features, hiperparâmetros, treinamento ou os arquivos dos modelos.
