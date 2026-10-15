"""Pagina de desempenho historico dos modelos."""

from __future__ import annotations

import streamlit as st

from src.dashboard.charts import performance_chart, residual_chart
from src.dashboard.components import (
    disclaimer,
    page_header,
    performance_card,
    section_header,
)
from src.dashboard.config import HORIZONS, TEST_METRICS
from src.dashboard.data_loader import load_model_comparison, load_test_predictions
from src.dashboard.performance_data import test_metrics_frame


page_header(
    "Avaliação histórica",
    "Desempenho dos modelos",
    "Métricas calculadas no conjunto de teste histórico. Elas descrevem o experimento e não o erro de previsões futuras em produção.",
)

section_header(
    "Métricas por horizonte",
    "MAE e RMSE são expressos em MW; MAPE representa o erro percentual absoluto médio.",
)
columns = st.columns(3)
for column, horizon in zip(columns, HORIZONS):
    with column:
        performance_card(horizon, TEST_METRICS[horizon])

metrics = test_metrics_frame()
st.plotly_chart(
    performance_chart(metrics),
    width="stretch",
    config={"displayModeBar": False},
)

section_header(
    "Comparação de modelos",
    "Resultados adicionais só são mostrados quando existe um artefato de avaliação persistido no projeto.",
)
comparison = load_model_comparison()
if comparison is None:
    st.info(
        "Não foi encontrado um artefato estruturado de comparação entre algoritmos. "
        "Por isso, nenhuma classificação adicional é apresentada."
    )
else:
    st.dataframe(comparison, width="stretch", hide_index=True)

section_header(
    "Análise de resíduos",
    "Resíduos de teste exigem valores observados e previstos pareados; previsões atuais ainda sem realização não entram nesta análise.",
)
predictions = load_test_predictions()
required = {"timestamp", "horizon", "actual_load_mw", "predicted_load_mw"}
if predictions is None:
    st.info(
        "Não foi encontrado um artefato de previsões do conjunto de teste. "
        "A aplicação não sintetiza resíduos nem usa previsões futuras como se fossem erros observados."
    )
elif not required.issubset(predictions.columns):
    st.warning(
        "O artefato de teste existe, mas não contém as colunas necessárias para a análise de resíduos: "
        + ", ".join(sorted(required - set(predictions.columns)))
    )
else:
    st.plotly_chart(
        residual_chart(predictions),
        width="stretch",
        config={"displayModeBar": False},
    )

disclaimer()
