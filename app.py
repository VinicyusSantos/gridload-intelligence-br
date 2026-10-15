"""Entrada da aplicacao de previsao de carga eletrica do Nordeste."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components import load_styles
from src.dashboard.data_loader import DashboardDataError, load_latest_forecasts
from src.dashboard.formatting import format_datetime


st.set_page_config(
    page_title="Previsão de carga elétrica do subsistema Nordeste",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_styles()

with st.sidebar:
    st.markdown(
        '<div class="grid-brand">Previsão de carga elétrica do subsistema Nordeste</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="grid-brand-subtitle">Previsão e interpretação da carga elétrica no subsistema Nordeste.</div>',
        unsafe_allow_html=True,
    )

navigation = st.navigation(
    [
        st.Page("pages/overview.py", title="Visão Geral", icon=":material/dashboard:", default=True),
        st.Page("pages/explainability.py", title="Explicabilidade", icon=":material/psychology:"),
        st.Page(
            "pages/performance.py",
            title="Análise dos Modelos",
            icon=":material/monitoring:",
        ),
    ],
    position="sidebar",
)

with st.sidebar:
    st.markdown("---")
    try:
        sidebar_forecasts = load_latest_forecasts()
        sidebar_updated = sidebar_forecasts["generated_at_utc"].iloc[0]
        st.caption(f"Última atualização: {format_datetime(sidebar_updated)} UTC")
    except (DashboardDataError, KeyError, IndexError):
        st.caption("Última atualização: indisponível")
    st.caption("Modelos XGBoost · horizontes de 1h, 6h e 24h")

navigation.run()
