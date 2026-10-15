"""Graficos Plotly com a identidade visual da aplicacao."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.config import HORIZON_COLORS, HORIZONS


INK = "#172033"
MUTED = "#64748B"
GRID = "#E2E8F0"


def _layout(figure: go.Figure, *, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, ui-sans-serif, system-ui", color=INK, size=12),
        hoverlabel=dict(bgcolor="#111827", font_color="#FFFFFF"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    figure.update_xaxes(showgrid=False, linecolor=GRID, tickfont_color=MUTED)
    figure.update_yaxes(gridcolor=GRID, zeroline=False, tickfont_color=MUTED)
    return figure


def recent_load_chart(
    history: pd.DataFrame,
    snapshot: dict[str, Any],
    *,
    hours: int = 48,
) -> go.Figure:
    recent = history.copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], errors="raise")
    recent = recent.sort_values("timestamp").tail(hours)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent["timestamp"],
            y=recent["load_mw"],
            name="Carga observada",
            mode="lines",
            line=dict(color="#334155", width=2.2),
            hovertemplate="%{x|%d/%m %H:%M}<br><b>%{y:,.0f} MW</b><extra></extra>",
        )
    )
    for horizon in HORIZONS:
        item = snapshot["forecasts"][horizon]
        figure.add_trace(
            go.Scatter(
                x=[item["target_timestamp"]],
                y=[item["forecast_mw"]],
            name=f"Previsão {horizon}",
                mode="markers",
                marker=dict(
                    color=HORIZON_COLORS[horizon],
                    size=13,
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=2),
                ),
                hovertemplate=(
                    f"{horizon} · %{{x|%d/%m %H:%M}}<br>"
                    "<b>%{y:,.0f} MW</b><extra></extra>"
                ),
            )
        )

    figure.update_yaxes(title="Carga (MW)")
    figure.update_xaxes(title=None)
    return _layout(figure, height=410)


def local_shap_chart(explanation: pd.DataFrame, *, top_n: int = 8) -> go.Figure:
    selected = explanation.nlargest(top_n, "abs_impact_mw").sort_values("impact_mw")
    colors = ["#2563EB" if value >= 0 else "#EA580C" for value in selected["impact_mw"]]
    figure = go.Figure(
        go.Bar(
            x=selected["impact_mw"],
            y=selected["feature"],
            orientation="h",
            marker_color=colors,
            customdata=selected["direction"],
            hovertemplate=(
                "%{y}<br><b>%{x:+.1f} MW</b><br>%{customdata}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#94A3B8")
    figure.update_xaxes(title="Contribuição para a previsão (MW)")
    figure.update_yaxes(title=None)
    return _layout(figure, height=430)


def global_shap_chart(global_shap: pd.DataFrame, *, top_n: int = 10) -> go.Figure:
    selected = global_shap.nlargest(top_n, "mean_abs_shap_mw").sort_values(
        "mean_abs_shap_mw"
    )
    figure = go.Figure(
        go.Bar(
            x=selected["mean_abs_shap_mw"],
            y=selected["feature"],
            orientation="h",
            marker_color="#0F766E",
            hovertemplate="%{y}<br><b>%{x:,.1f} MW</b><extra></extra>",
        )
    )
    figure.update_xaxes(title="Impacto médio absoluto (SHAP, MW)")
    figure.update_yaxes(title=None)
    return _layout(figure, height=450)


def performance_chart(metrics: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.36],
        subplot_titles=("Erros em MW", "Erro percentual"),
    )
    figure.add_trace(
        go.Bar(
            x=metrics["horizon"],
            y=metrics["mae_mw"],
            name="MAE",
            marker_color="#2563EB",
            hovertemplate="%{x}<br>MAE: <b>%{y:,.2f} MW</b><extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=metrics["horizon"],
            y=metrics["rmse_mw"],
            name="RMSE",
            marker_color="#7C3AED",
            hovertemplate="%{x}<br>RMSE: <b>%{y:,.2f} MW</b><extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=metrics["horizon"],
            y=metrics["mape_pct"],
            name="MAPE",
            marker_color="#0F766E",
            hovertemplate="%{x}<br>MAPE: <b>%{y:.2f}%</b><extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title="MW", row=1, col=1)
    figure.update_yaxes(title="%", row=1, col=2)
    figure.update_layout(barmode="group")
    return _layout(figure, height=430)


def residual_chart(predictions: pd.DataFrame) -> go.Figure:
    frame = predictions.copy()
    frame["residual_mw"] = frame["actual_load_mw"] - frame["predicted_load_mw"]
    figure = go.Figure()
    for horizon, subset in frame.groupby("horizon", sort=False):
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(subset["timestamp"]),
                y=subset["residual_mw"],
                mode="markers",
                marker=dict(color=HORIZON_COLORS.get(str(horizon), MUTED), size=5),
                name=str(horizon),
                hovertemplate="%{x|%d/%m/%Y %H:%M}<br>%{y:+.1f} MW<extra></extra>",
            )
        )
    figure.add_hline(y=0, line_width=1, line_color="#94A3B8")
    figure.update_yaxes(title="Real - previsto (MW)")
    return _layout(figure, height=390)


def baseline_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    """Compara o MAE final do XGBoost ao melhor baseline de cada horizonte."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=comparison["horizon"],
            y=comparison["baseline_mae_mw"],
            name="Melhor baseline",
            marker_color="#94A3B8",
            customdata=comparison[["best_baseline"]],
            hovertemplate=(
                "%{x}<br>%{customdata[0]}<br>MAE: <b>%{y:,.2f} MW</b>"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=comparison["horizon"],
            y=comparison["model_mae_mw"],
            name="XGBoost tunado",
            marker_color="#2563EB",
            customdata=comparison[["mae_reduction_pct"]],
            hovertemplate=(
                "%{x}<br>MAE: <b>%{y:,.2f} MW</b><br>"
                "Redução: %{customdata[0]:.1f}%<extra></extra>"
            ),
        )
    )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title="MAE (MW)")
    return _layout(figure, height=410)


def validation_model_chart(frame: pd.DataFrame, metric: str) -> go.Figure:
    """Compara familias de modelos no mesmo conjunto de validacao."""
    labels = {
        "mae_mw": ("MAE (MW)", ",.2f"),
        "rmse_mw": ("RMSE (MW)", ",.2f"),
        "mape_pct": ("MAPE (%)", ".2f"),
    }
    if metric not in labels:
        raise ValueError(f"Metrica de validacao desconhecida: {metric}")

    title, number_format = labels[metric]
    colors = {
        "Regressao linear": "#94A3B8",
        "Random Forest": "#0F766E",
        "XGBoost base": "#7C3AED",
    }
    figure = go.Figure()
    for model in ("Regressao linear", "Random Forest", "XGBoost base"):
        subset = frame.loc[frame["model"] == model]
        figure.add_trace(
            go.Bar(
                x=subset["horizon"],
                y=subset[metric],
                name=model,
                marker_color=colors[model],
                customdata=subset[["feature_set"]],
                hovertemplate=(
                    "%{x}<br>%{customdata[0]}<br>"
                    f"{title}: <b>%{{y:{number_format}}}</b><extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title=title)
    return _layout(figure, height=420)


def tuning_history_chart(frame: pd.DataFrame, metric: str) -> go.Figure:
    """Mostra validacao, tuning e teste final sem ocultar a troca de particao."""
    labels = {
        "mae_mw": "MAE (MW)",
        "rmse_mw": "RMSE (MW)",
        "mape_pct": "MAPE (%)",
    }
    if metric not in labels:
        raise ValueError(f"Metrica de tuning desconhecida: {metric}")

    figure = go.Figure()
    for horizon in HORIZONS:
        subset = frame.loc[frame["horizon"] == horizon].sort_values("stage")
        figure.add_trace(
            go.Scatter(
                x=subset["stage"].astype(str),
                y=subset[metric],
                mode="lines+markers",
                name=horizon,
                line=dict(color=HORIZON_COLORS[horizon], width=2.5),
                marker=dict(size=9),
                customdata=subset[["split"]],
                hovertemplate=(
                    "%{x}<br>%{customdata[0]}<br>"
                    f"{labels[metric]}: <b>%{{y:.2f}}</b><extra></extra>"
                ),
            )
        )
        final = subset.loc[subset["split"] == "Teste"]
        figure.add_trace(
            go.Scatter(
                x=final["stage"].astype(str),
                y=final[metric],
                mode="markers",
                marker=dict(
                    color=HORIZON_COLORS[horizon],
                    size=13,
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=2),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.update_yaxes(title=labels[metric])
    return _layout(figure, height=430)


def model_diagnostics_chart(metrics: pd.DataFrame) -> go.Figure:
    """Resume qualidade de ajuste, vies e cauda de erro do teste final."""
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("R²", "Viés observado - previsto", "Maior erro absoluto"),
    )
    colors = [HORIZON_COLORS[horizon] for horizon in metrics["horizon"]]
    figure.add_trace(
        go.Bar(
            x=metrics["horizon"],
            y=metrics["r2"],
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>R²: <b>%{y:.3f}</b><extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=metrics["horizon"],
            y=metrics["bias_mw"],
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>Viés: <b>%{y:+.2f} MW</b><extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=metrics["horizon"],
            y=metrics["max_abs_error_mw"],
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>Erro: <b>%{y:,.2f} MW</b><extra></extra>",
        ),
        row=1,
        col=3,
    )
    figure.add_hline(y=0, line_width=1, line_color="#94A3B8", row=1, col=2)
    figure.update_yaxes(range=[0, 1], row=1, col=1)
    figure.update_yaxes(title="MW", row=1, col=2)
    figure.update_yaxes(title="MW", row=1, col=3)
    return _layout(figure, height=410)
