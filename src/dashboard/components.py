"""Componentes visuais reutilizaveis do dashboard."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import streamlit as st

from src.dashboard.config import DISCLAIMER, HORIZON_COLORS, STYLE_PATH
from src.dashboard.formatting import (
    format_datetime,
    format_mw,
    format_percent,
    trend_symbol,
)


def load_styles() -> None:
    if STYLE_PATH.is_file():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        (
            f'<div class="grid-eyebrow">{escape(eyebrow)}</div>'
            f'<h1 class="grid-title">{escape(title)}</h1>'
            f'<p class="grid-subtitle">{escape(subtitle)}</p>'
        ),
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    text = f'<div class="grid-section"><h2>{escape(title)}</h2>'
    if description:
        text += f"<p>{escape(description)}</p>"
    text += "</div>"
    st.markdown(text, unsafe_allow_html=True)


def current_load_card(snapshot: dict[str, Any]) -> None:
    delta_mw = snapshot.get("current_delta_mw")
    delta_pct = snapshot.get("current_delta_pct")
    st.markdown(
        f"""
        <div class="grid-card grid-card-current">
          <div class="grid-card-label">Carga observada mais recente</div>
          <div class="grid-card-value">{escape(format_mw(snapshot['current_load_mw']))}</div>
          <div class="grid-card-delta" style="--accent: #334155">
            {trend_symbol(delta_mw)} {escape(format_mw(delta_mw, signed=True))}
            · {escape(format_percent(delta_pct, signed=True))} na última hora
          </div>
          <div class="grid-card-meta">Referência: {escape(format_datetime(snapshot['current_timestamp']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def forecast_card(item: dict[str, Any]) -> None:
    horizon = str(item["horizon"])
    color = HORIZON_COLORS.get(horizon, "#2563EB")
    status = item.get("historical_context_status")
    percentile = item.get("percentile")
    if status == "available" and percentile is not None and np.isfinite(percentile):
        marker = min(100.0, max(0.0, float(percentile)))
        percentile_html = f"""
          <div class="grid-percentile-label">{escape(str(item.get('percentile_label') or ''))}</div>
          <div class="grid-percentile-track">
            <span class="grid-percentile-marker" style="left:{marker:.2f}%"></span>
          </div>
          <div class="grid-percentile-scale"><span>P0</span><span>P25</span><span>P50</span><span>P75</span><span>P100</span></div>
          <div class="grid-card-meta grid-reference">Comparado com {escape(str(item.get('historical_reference') or 'referência indisponível'))}.</div>
          <div class="grid-card-meta grid-context">Maior que aproximadamente {escape(format_percent(percentile))} das cargas historicamente observadas nessa mesma referência.</div>
        """
    else:
        reason = item.get("historical_context_reason") or "Amostra comparável insuficiente."
        percentile_html = (
            '<div class="grid-card-meta grid-context">Contexto histórico indisponível: '
            f"{escape(str(reason))}</div>"
        )

    st.markdown(
        f"""
        <div class="grid-card grid-card-accent" style="--accent:{color}">
          <div class="grid-card-label">Previsão +{escape(horizon)}</div>
          <div class="grid-card-value">{escape(format_mw(item['forecast_mw']))}</div>
          <div class="grid-card-delta" style="--accent:{color}">
            {trend_symbol(item['delta_mw'])} {escape(format_mw(item['delta_mw'], signed=True))}
            · {escape(format_percent(item['delta_pct'], signed=True))} vs. carga atual
          </div>
          <div class="grid-card-meta">Alvo: {escape(format_datetime(item['target_timestamp']))}</div>
          {percentile_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def performance_card(horizon: str, metrics: dict[str, float]) -> None:
    color = HORIZON_COLORS.get(horizon, "#2563EB")
    st.markdown(
        f"""
        <div class="grid-card grid-card-accent" style="--accent:{color}">
          <div class="grid-card-label">Horizonte {escape(horizon)}</div>
          <div class="grid-card-value">{escape(format_percent(metrics['mape_pct'], 2))}</div>
          <div class="grid-card-meta">MAPE no conjunto de teste</div>
          <div class="grid-kpi-row"><span>MAE</span><strong>{escape(format_mw(metrics['mae_mw'], 2))}</strong></div>
          <div class="grid-kpi-row"><span>RMSE</span><strong>{escape(format_mw(metrics['rmse_mw'], 2))}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str, *, summary: bool = False) -> None:
    css_class = "grid-summary" if summary else "grid-note"
    st.markdown(f'<div class="{css_class}">{escape(text)}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(
        f'<div class="grid-disclaimer">{escape(DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )
