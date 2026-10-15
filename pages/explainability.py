"""Pagina de explicabilidade local e global dos modelos."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import global_shap_chart, local_shap_chart
from src.dashboard.components import (
    disclaimer,
    forecast_card,
    page_header,
    section_header,
)
from src.dashboard.config import HORIZONS
from src.dashboard.data_loader import DashboardDataError, load_global_shap
from src.dashboard.explainability import local_explanation
from src.dashboard.forecast_service import get_current_forecast
from src.dashboard.formatting import format_mw, format_number_br


page_header(
    "Transparência do modelo",
    "Por que o modelo chegou a esta previsão?",
    "As contribuições SHAP mostram como cada grupo de informações deslocou a resposta do modelo, sem implicar causalidade.",
)

horizon = st.segmented_control(
    "Horizonte analisado",
    options=list(HORIZONS),
    default="1h",
    selection_mode="single",
)
if horizon is None:
    horizon = "1h"

try:
    snapshot = get_current_forecast()
except (DashboardDataError, FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Não foi possível carregar a previsão atual: {error}")
    disclaimer()
    st.stop()

section_header("Previsão em foco")
forecast_card(snapshot["forecasts"][horizon])

section_header(
    "Explicação local",
    "Fatores que mais elevaram ou reduziram a previsão atual em relação ao valor-base interno do modelo.",
)
st.markdown(
    '<div class="grid-legend"><span><i class="grid-dot" style="background:#2563EB"></i>Eleva a previsão</span>'
    '<span><i class="grid-dot" style="background:#EA580C"></i>Reduz a previsão</span></div>',
    unsafe_allow_html=True,
)
try:
    with st.spinner("Calculando contribuições SHAP da previsão atual..."):
        local = local_explanation(horizon)
    st.plotly_chart(
        local_shap_chart(local, top_n=8),
        width="stretch",
        config={"displayModeBar": False},
    )
    strongest = local.iloc[0]
    st.caption(
        f"Maior contribuição absoluta: {strongest['feature']} "
        f"({format_mw(strongest['impact_mw'], 1, signed=True)}). "
        "Valores são contribuições do modelo, não efeitos causais."
    )
except (FileNotFoundError, ValueError, KeyError) as error:
    st.warning(f"A explicação local não está disponível: {error}")

section_header(
    "Importância global",
    "Impacto médio absoluto observado em uma amostra temporal da base de modelagem.",
)
global_artifact = load_global_shap()
if global_artifact is None:
    st.info(
        "O artefato SHAP global ainda não foi gerado. Execute "
        "`python -m src.explainability.build_shap_artifacts` para criá-lo com os dados e modelos reais."
    )
else:
    selected = global_artifact.loc[global_artifact["horizon"] == horizon].copy()
    required = {"feature", "mean_abs_shap_mw", "sample_size", "generated_at_utc"}
    if selected.empty or not required.issubset(selected.columns):
        st.warning("O artefato SHAP global não possui dados válidos para este horizonte.")
    else:
        st.plotly_chart(
            global_shap_chart(selected, top_n=10),
            width="stretch",
            config={"displayModeBar": False},
        )
        generated = pd.Timestamp(selected["generated_at_utc"].iloc[0]).strftime("%d/%m/%Y %H:%M UTC")
        st.caption(
            f"Artefato calculado com {format_number_br(selected['sample_size'].iloc[0])} observações; gerado em {generated}. "
            "Componentes cíclicas de horário, dia da semana e época do ano foram agregadas por conceito."
        )

disclaimer()
