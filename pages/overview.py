"""Pagina de visao geral das previsoes atuais."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import recent_load_chart
from src.dashboard.components import (
    current_load_card,
    disclaimer,
    forecast_card,
    note,
    page_header,
    section_header,
)
from src.dashboard.config import HORIZONS, RECENT_HISTORY_HOURS
from src.dashboard.data_loader import DashboardDataError, load_base_data
from src.dashboard.forecast_service import forecast_summary, get_current_forecast
from src.dashboard.formatting import format_datetime


page_header(
    "Monitoramento preditivo",
    "Previsão de carga elétrica do subsistema Nordeste",
    "Previsão de carga elétrica do subsistema Nordeste com Machine Learning",
)

action_col, status_col = st.columns([0.24, 0.76], vertical_alignment="center")
with action_col:
    refresh = st.button(
        "Atualizar previsões",
        type="primary",
        width="stretch",
        help="Recalcula a inferência com os modelos e dados locais. Não baixa dados nem retreina os modelos.",
    )

if refresh:
    try:
        with st.spinner("Aplicando os modelos persistidos..."):
            get_current_forecast(refresh=True)
        st.session_state["forecast_refresh_success"] = True
        st.rerun()
    except (DashboardDataError, FileNotFoundError, ValueError, KeyError) as error:
        st.error(f"Não foi possível atualizar as previsões: {error}")

try:
    snapshot = get_current_forecast()
except (DashboardDataError, FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Não foi possível carregar a previsão atual: {error}")
    disclaimer()
    st.stop()

if snapshot.get("recalculated"):
    st.session_state["forecast_auto_update_notice"] = True
    st.rerun()

with status_col:
    generated = snapshot.get("generated_at")
    generated_label = format_datetime(generated)
    timezone_label = " UTC" if isinstance(generated, pd.Timestamp) and generated.tzinfo else ""
    st.markdown(
        f'<span class="grid-status">Atualizado em {generated_label}{timezone_label}</span>',
        unsafe_allow_html=True,
    )

if st.session_state.pop("forecast_refresh_success", False):
    st.success("Previsões recalculadas com sucesso. Nenhum modelo foi retreinado.")
elif st.session_state.pop("forecast_auto_update_notice", False):
    st.info("O artefato de previsão estava ausente ou desatualizado e foi recalculado com os modelos existentes.")

section_header("Situação atual", "Variação calculada contra a observação da hora anterior.")
current_load_card(snapshot)

section_header(
    "Próximos horizontes",
    "Cada cartão compara a previsão com a carga atual e a posiciona em seu contexto histórico equivalente.",
)
columns = st.columns(3)
for column, horizon in zip(columns, HORIZONS):
    with column:
        forecast_card(snapshot["forecasts"][horizon])

section_header(
    "Trajetória recente e pontos previstos",
    "A linha representa apenas valores observados. Os losangos são previsões isoladas e não formam uma curva interpolada.",
)
history = load_base_data()
st.plotly_chart(
    recent_load_chart(history, snapshot, hours=RECENT_HISTORY_HOURS),
    width="stretch",
    config={"displayModeBar": False},
)

section_header("Leitura automática")
note(forecast_summary(snapshot), summary=True)
disclaimer()
