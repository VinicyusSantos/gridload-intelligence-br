"""EDA e desempenho historico dos modelos finais tunados."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    baseline_comparison_chart,
    model_diagnostics_chart,
    performance_chart,
    residual_chart,
    tuning_history_chart,
    validation_model_chart,
)
from src.dashboard.components import (
    disclaimer,
    note,
    page_header,
    performance_card,
    section_header,
)
from src.dashboard.config import HORIZONS, TEST_METRICS
from src.dashboard.data_loader import (
    DashboardDataError,
    load_json,
    load_model_comparison,
    load_test_predictions,
)
from src.dashboard.formatting import (
    format_datetime,
    format_mw,
    format_number_br,
    format_percent,
)
from src.dashboard.performance_data import (
    EXPERIMENT_PERIOD,
    baseline_metrics_frame,
    best_baseline_comparison_frame,
    model_configuration_frame,
    test_metrics_frame,
    test_baseline_metrics_frame,
    tuning_metrics_frame,
    validation_metrics_frame,
    worst_cases_frame,
)
from src.models.train_models import METADATA_FILE, load_model_specs


CHART_CONFIG = {"displayModeBar": False}
METRIC_OPTIONS = {"MAE": "mae_mw", "RMSE": "rmse_mw", "MAPE": "mape_pct"}


def _metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "Horizonte": frame["horizon"],
            "Features": frame["feature_count"].map(lambda value: format_number_br(value)),
            "MAE": frame["mae_mw"].map(lambda value: format_mw(value, 2)),
            "RMSE": frame["rmse_mw"].map(lambda value: format_mw(value, 2)),
            "MAPE": frame["mape_pct"].map(lambda value: format_percent(value, 2)),
            "R²": frame["r2"].map(lambda value: format_number_br(value, 3)),
            "Viés": frame["bias_mw"].map(
                lambda value: format_mw(value, 2, signed=True)
            ),
            "Maior erro": frame["max_abs_error_mw"].map(
                lambda value: format_mw(value, 2)
            ),
        }
    )
    return result


def _baseline_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Horizonte": frame["horizon"],
            "Referência": frame["reference"],
            "MAE": frame["mae_mw"].map(lambda value: format_mw(value, 2)),
            "RMSE": frame["rmse_mw"].map(lambda value: format_mw(value, 2)),
            "MAPE": frame["mape_pct"].map(lambda value: format_percent(value, 2)),
        }
    )


def _validation_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Horizonte": frame["horizon"],
            "Modelo": frame["model"],
            "Conjunto de features": frame["feature_set"],
            "MAE": frame["mae_mw"].map(lambda value: format_mw(value, 2)),
            "RMSE": frame["rmse_mw"].map(lambda value: format_mw(value, 2)),
            "MAPE": frame["mape_pct"].map(lambda value: format_percent(value, 2)),
        }
    )


page_header(
    "EDA dos modelos finais",
    "Análise dos modelos XGBoost tunados",
    "Do baseline ao teste final: desempenho, tuning, erros extremos, features e configuração operacional dos horizontes de 1h, 6h e 24h.",
)

st.info(
    "Esta página descreve o experimento temporal preservado nos notebooks. "
    f"Foram {format_number_br(EXPERIMENT_PERIOD['rows'])} observações, com "
    f"{format_number_br(EXPERIMENT_PERIOD['test_rows'])} horas no teste entre "
    f"{pd.Timestamp(EXPERIMENT_PERIOD['test_start']).strftime('%d/%m/%Y %Hh')} e "
    f"{pd.Timestamp(EXPERIMENT_PERIOD['test_end']).strftime('%d/%m/%Y %Hh')}. "
    "Essas métricas não são uma avaliação dos modelos posteriormente "
    "retreinados com toda a base."
)

summary_tab, evolution_tab, diagnostics_tab, configuration_tab = st.tabs(
    [
        "Resumo final",
        "Evolução da modelagem",
        "Diagnóstico de erros",
        "Configuração final",
    ]
)

metrics = test_metrics_frame()

with summary_tab:
    section_header(
        "Métricas no teste final",
        "MAE e RMSE estão em MW; MAPE é o erro percentual absoluto médio. R² mede a parcela da variabilidade explicada.",
    )
    columns = st.columns(3)
    for column, horizon in zip(columns, HORIZONS):
        with column:
            performance_card(horizon, TEST_METRICS[horizon])

    st.plotly_chart(
        performance_chart(metrics),
        width="stretch",
        config=CHART_CONFIG,
    )
    st.dataframe(_metrics_table(metrics), width="stretch", hide_index=True)
    note(
        "O horizonte de 1h é o mais preciso (MAPE de 1,14% e R² de 0,976). "
        "Em 6h e 24h a precisão diminui, como esperado, e a cauda de erro cresce; "
        "o maior erro absoluto aparece em 24h."
    )

with evolution_tab:
    section_header(
        "Ganho sobre regras simples",
        "As regras do notebook de baseline foram recalculadas nas mesmas 6.106 horas do teste final, permitindo uma comparação temporal equivalente.",
    )
    baseline_comparison = best_baseline_comparison_frame()
    st.plotly_chart(
        baseline_comparison_chart(baseline_comparison),
        width="stretch",
        config=CHART_CONFIG,
    )
    gains = " · ".join(
        f"{row.horizon}: {format_percent(row.mae_reduction_pct, 1)}"
        for row in baseline_comparison.itertuples(index=False)
    )
    st.caption(f"Redução de MAE contra o melhor baseline — {gains}.")

    with st.expander("Ver todos os baselines no conjunto de teste"):
        st.dataframe(
            _baseline_table(test_baseline_metrics_frame()),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Ver resultados originais do notebook 05"):
        st.caption(
            "Estes valores foram calculados no notebook sobre as 40.704 horas "
            "completas e, por isso, não são usados no percentual de ganho acima."
        )
        st.dataframe(
            _baseline_table(baseline_metrics_frame()),
            width="stretch",
            hide_index=True,
        )

    section_header(
        "Comparação na validação",
        "Regressão linear, Random Forest e XGBoost foram comparados na mesma partição temporal. O melhor conjunto de features de cada família é exibido.",
    )
    validation = validation_metrics_frame()
    selected_validation_metric = st.segmented_control(
        "Métrica da comparação",
        options=list(METRIC_OPTIONS),
        default="MAE",
        key="validation_metric",
    ) or "MAE"
    st.plotly_chart(
        validation_model_chart(
            validation,
            METRIC_OPTIONS[selected_validation_metric],
        ),
        width="stretch",
        config=CHART_CONFIG,
    )
    st.dataframe(_validation_table(validation), width="stretch", hide_index=True)

    section_header(
        "Trajetória do XGBoost",
        "As três primeiras etapas usam validação. O losango final usa o conjunto de teste, portanto não deve ser interpretado como uma quarta otimização.",
    )
    tuning = tuning_metrics_frame()
    selected_tuning_metric = st.segmented_control(
        "Métrica da trajetória",
        options=list(METRIC_OPTIONS),
        default="MAE",
        key="tuning_metric",
    ) or "MAE"
    st.plotly_chart(
        tuning_history_chart(tuning, METRIC_OPTIONS[selected_tuning_metric]),
        width="stretch",
        config=CHART_CONFIG,
    )
    st.caption(
        "Busca: TimeSeriesSplit com 5 divisões e RandomizedSearchCV com 50 "
        "combinações. O teste permaneceu fora da seleção de hiperparâmetros."
    )

    additional_comparison = load_model_comparison()
    if additional_comparison is not None:
        with st.expander("Artefato adicional de comparação"):
            st.dataframe(additional_comparison, width="stretch", hide_index=True)

with diagnostics_tab:
    section_header(
        "Perfil dos erros",
        "Viés positivo significa que, em média, o valor observado ficou acima do previsto. O erro máximo mostra a cauda, não o comportamento típico.",
    )
    st.plotly_chart(
        model_diagnostics_chart(metrics),
        width="stretch",
        config=CHART_CONFIG,
    )

    section_header(
        "Piores casos do teste",
        "Maior erro absoluto registrado em cada horizonte no notebook de explicabilidade.",
    )
    worst = worst_cases_frame()
    worst_display = pd.DataFrame(
        {
            "Horizonte": worst["horizon"],
            "Data": worst["timestamp"].map(format_datetime),
            "Observado": worst["actual_load_mw"].map(lambda value: format_mw(value, 2)),
            "Previsto": worst["predicted_load_mw"].map(
                lambda value: format_mw(value, 2)
            ),
            "Erro absoluto": worst["absolute_error_mw"].map(
                lambda value: format_mw(value, 2)
            ),
            "Direção": worst["direction"],
        }
    )
    st.dataframe(worst_display, width="stretch", hide_index=True)
    note(
        "Os vieses médios são pequenos, mas os piores casos de 6h e 24h passam "
        "de 2,6 GW. Isso reforça o monitoramento separado de feriados, rampas e "
        "eventos atípicos, além das métricas médias."
    )

    section_header(
        "Resíduos horários",
        "A série completa requer valores observados e previstos pareados do conjunto de teste.",
    )
    predictions = load_test_predictions()
    required = {"timestamp", "horizon", "actual_load_mw", "predicted_load_mw"}
    if predictions is None:
        st.info(
            "O resumo quantitativo e os piores casos estão disponíveis acima, "
            "mas `data/evaluation/test_predictions.parquet` ainda não existe. "
            "Por isso, a aplicação não inventa uma distribuição de resíduos."
        )
    elif not required.issubset(predictions.columns):
        st.warning(
            "O artefato de teste não contém as colunas necessárias: "
            + ", ".join(sorted(required - set(predictions.columns)))
        )
    else:
        st.plotly_chart(
            residual_chart(predictions),
            width="stretch",
            config=CHART_CONFIG,
        )

with configuration_tab:
    section_header(
        "Hiperparâmetros persistidos",
        "A tabela é lida diretamente de params.json e features.json, os mesmos arquivos usados pelo treinamento do pipeline.",
    )
    configuration = model_configuration_frame().rename(
        columns={
            "horizon": "Horizonte",
            "feature_count": "Features",
            "n_estimators": "Árvores",
            "learning_rate": "Learning rate",
            "max_depth": "Profundidade",
            "min_child_weight": "Min child weight",
            "subsample": "Subsample",
            "colsample_bytree": "Colsample",
            "gamma": "Gamma",
            "reg_alpha": "L1",
            "reg_lambda": "L2",
        }
    )
    st.dataframe(configuration, width="stretch", hide_index=True)

    section_header(
        "Features por horizonte",
        "A seleção muda conforme a antecedência da previsão.",
    )
    for spec in load_model_specs():
        with st.expander(f"{spec.horizon} · {len(spec.features)} features"):
            st.code("\n".join(spec.features), language="text")

    section_header(
        "Modelo operacional atual",
        "Depois do experimento, o pipeline reutiliza a configuração e retreina os modelos com todas as linhas rotuladas disponíveis.",
    )
    try:
        metadata = load_json(METADATA_FILE)
        metadata_columns = st.columns(3)
        metadata_columns[0].metric(
            "Linhas de treinamento",
            format_number_br(metadata.get("training_rows")),
        )
        metadata_columns[1].metric(
            "Início",
            format_datetime(metadata.get("period_start")),
        )
        metadata_columns[2].metric(
            "Fim",
            format_datetime(metadata.get("period_end")),
        )
        st.caption(
            "Treinado em "
            f"{format_datetime(metadata.get('trained_at_utc'))} UTC · "
            f"XGBoost {metadata.get('xgboost_version', 'indisponível')}"
        )
    except (DashboardDataError, FileNotFoundError, ValueError, KeyError) as error:
        st.warning(f"Metadados do treinamento indisponíveis: {error}")

    note(
        "Os modelos operacionais usam toda a base rotulada. As métricas do "
        "teste acima continuam sendo evidência do experimento congelado e não "
        "uma nova medição fora da amostra dos JSON atuais."
    )

disclaimer()
