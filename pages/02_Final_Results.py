from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTCOME_LABELS = {
    "CAR[-3,-1]": "CAR_-3_-1",
    "CAR[-1,+3]": "CAR_-1_3",
    "CAR[+1,+60]": "CAR_1_60",
}

SPEC_LABEL_ORDER = [
    "Topic + severity",
    "Topic + severity + pre-volatility",
    "Topic + severity + industry",
    "Topic + severity + pre-volatility + industry",
]


st.set_page_config(page_title="Final Results", layout="wide")
st.title("Final Results")
st.caption("Official event-study and regression outputs from the current project pipeline.")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_regression_dataset() -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / "regression_dataset.parquet"
    csv_path = PROCESSED_DIR / "regression_dataset.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def outcome_description(outcome: str) -> str:
    descriptions = {
        "CAR_-3_-1": "Pre-event window. This checks whether prices move before the SEC event date.",
        "CAR_-1_3": "Immediate reaction window. This is the main short-window market response around disclosure.",
        "CAR_1_60": "Post-event drift window. This checks whether the market continues to react after the event.",
    }
    return descriptions[outcome]


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def build_outcome_regression_table(coefficients: pd.DataFrame, outcome_label: str) -> pd.DataFrame:
    selected = coefficients[
        (coefficients["outcome_label"] == outcome_label)
        & coefficients["term_group"].isin(["topic", "main_variable", "control"])
    ].copy()
    selected["term_group_order"] = selected["term_group"].map({"topic": 0, "main_variable": 1, "control": 2})
    selected["spec_label"] = pd.Categorical(selected["spec_label"], categories=SPEC_LABEL_ORDER, ordered=True)
    selected = selected.sort_values(["term_group_order", "term_label", "spec_label"])
    table = (
        selected.pivot_table(
            index=["term_group_order", "term_label"],
            columns="spec_label",
            values="coefficient_display",
            aggfunc="first",
            fill_value="",
            observed=False,
        )
        .reset_index()
        .drop(columns=["term_group_order"])
    )
    table.columns.name = None
    return table.rename(columns={"term_label": "Variable"})


def build_outcome_diagnostics_table(model_summary: pd.DataFrame, outcome_label: str) -> pd.DataFrame:
    selected = model_summary[model_summary["outcome_label"] == outcome_label].copy()
    selected["spec_label"] = pd.Categorical(selected["spec_label"], categories=SPEC_LABEL_ORDER, ordered=True)
    selected = selected.sort_values("spec_label")
    diagnostics = selected[
        [
            "spec_label",
            "n_obs",
            "r_squared",
            "adj_r_squared",
            "include_pre_event_volatility",
            "include_industry",
        ]
    ].rename(
        columns={
            "spec_label": "Specification",
            "n_obs": "Observations",
            "r_squared": "R-squared",
            "adj_r_squared": "Adj. R-squared",
            "include_pre_event_volatility": "Pre-volatility",
            "include_industry": "Industry",
        }
    )
    diagnostics["R-squared"] = diagnostics["R-squared"].map(lambda value: f"{value:.3f}")
    diagnostics["Adj. R-squared"] = diagnostics["Adj. R-squared"].map(lambda value: f"{value:.3f}")
    diagnostics["Pre-volatility"] = diagnostics["Pre-volatility"].map(lambda value: "Yes" if value else "No")
    diagnostics["Industry"] = diagnostics["Industry"].map(lambda value: "Yes" if value else "No")
    return diagnostics


topic_summary_path = TABLES_DIR / "topic_car_summary.csv"
topic_average_car_path = TABLES_DIR / "topic_average_car.csv"
average_car_path = TABLES_DIR / "average_car.csv"
regression_coefficients_path = TABLES_DIR / "regression_coefficients_long.csv"
regression_model_summary_path = TABLES_DIR / "regression_model_summary.csv"
regression_dataset_path = PROCESSED_DIR / "regression_dataset.parquet"
topic_plot_path = FIGURES_DIR / "topic_average_car.png"
average_plot_path = FIGURES_DIR / "average_car_plot.png"

required = [
    topic_summary_path,
    topic_average_car_path,
    average_car_path,
    regression_coefficients_path,
    regression_model_summary_path,
    regression_dataset_path,
]
if not all(path.exists() for path in required):
    st.warning("Official result artifacts are not available yet. Run `./.venv/bin/python run.py --step event_study`.")
else:
    topic_summary = read_csv(topic_summary_path)
    topic_average_car = read_csv(topic_average_car_path)
    average_car = read_csv(average_car_path)
    regression_coefficients = read_csv(regression_coefficients_path)
    regression_model_summary = read_csv(regression_model_summary_path)
    reg_df = read_regression_dataset()
    immediate_high = topic_summary.loc[topic_summary["mean_CAR_immediate"].idxmax()]
    immediate_low = topic_summary.loc[topic_summary["mean_CAR_immediate"].idxmin()]
    drift_high = topic_summary.loc[topic_summary["mean_CAR_drift"].idxmax()]
    drift_low = topic_summary.loc[topic_summary["mean_CAR_drift"].idxmin()]
    severity_sig_count = int(
        (
            (regression_coefficients["term_label"] == "Severity score")
            & (regression_coefficients["p_value"] < 0.10)
        ).sum()
    )
    topic_sig_count = int(
        (
            (regression_coefficients["term_group"] == "topic")
            & (regression_coefficients["p_value"] < 0.10)
        ).sum()
    )
    max_r_squared = regression_model_summary["r_squared"].max()
    max_adj_r_squared = regression_model_summary["adj_r_squared"].max()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Events", reg_df["event_id"].nunique())
    metric_cols[1].metric("Regression included", int((reg_df["regression_inclusion_status"] == "included").sum()))
    metric_cols[2].metric("Topic groups", topic_summary["topic"].nunique())
    metric_cols[3].metric("Industries", reg_df["industry"].nunique())

    tabs = st.tabs(["Topic CAR", "Regression Playground"])

    with tabs[0]:
        st.header("Topic CAR")
        st.write("This view compares average cumulative abnormal returns across topic groups.")

        available_topics = sorted(topic_average_car["topic"].dropna().unique().tolist())
        summary_cols = [
            "topic",
            "n_events",
            "mean_CAR_pre",
            "mean_CAR_immediate",
            "mean_CAR_drift",
            "mean_severity",
        ]
        st.subheader("Topic CAR Summary")
        topic_summary_display = topic_summary[summary_cols].rename(
            columns={
                "mean_CAR_pre": "mean_CAR_-3_-1",
                "mean_CAR_immediate": "mean_CAR_-1_3",
                "mean_CAR_drift": "mean_CAR_1_60",
            }
        )
        for car_col in ["mean_CAR_-3_-1", "mean_CAR_-1_3", "mean_CAR_1_60"]:
            topic_summary_display[car_col] = topic_summary_display[car_col].map(lambda value: f"{value:.2%}")
        st.dataframe(topic_summary_display, use_container_width=True, hide_index=True)

        selected_topic = st.selectbox("Topic", ["All topics"] + available_topics)
        day_min, day_max = int(topic_average_car["relative_day"].min()), int(topic_average_car["relative_day"].max())
        default_window = (max(day_min, -10), day_max)
        selected_window = st.slider("Relative trading-day window", day_min, day_max, default_window)
        car_window_options = ["CAR_-3_-1", "CAR_-1_3", "CAR_1_60"]
        selected_shading_window = st.selectbox(
            "CAR window shading",
            ["All windows"] + car_window_options,
        )

        if selected_topic == "All topics":
            selected_topics = available_topics
        else:
            selected_topics = [selected_topic]

        filtered_topic_car = topic_average_car[
            topic_average_car["topic"].isin(selected_topics)
            & topic_average_car["relative_day"].between(selected_window[0], selected_window[1])
        ].copy().sort_values(["topic", "relative_day"])
        filtered_average_car = (
            average_car[average_car["relative_day"].between(selected_window[0], selected_window[1])]
            .copy()
            .sort_values("relative_day")
        )
        filtered_average_car["average_CAR_display"] = filtered_average_car["average_abnormal_return"].cumsum()
        filtered_topic_car["average_CAR_display"] = filtered_topic_car.groupby("topic")[
            "average_abnormal_return"
        ].cumsum()
        st.caption(
            "Charts reset cumulative CAR at the selected window start by cumulatively summing average abnormal returns."
        )

        st.subheader("All Events")
        all_events_line = (
            alt.Chart(filtered_average_car)
            .mark_line(color="#1f77b4", strokeWidth=2.5)
            .encode(
                x=alt.X("relative_day:Q", title="Relative trading day"),
                y=alt.Y("average_CAR_display:Q", title="Average CAR", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("relative_day:Q", title="Relative day"),
                    alt.Tooltip("average_CAR_display:Q", title="Average CAR", format=".2%"),
                    alt.Tooltip("average_abnormal_return:Q", title="Average abnormal return", format=".2%"),
                    alt.Tooltip("n_events:Q", title="Events"),
                ],
            )
        )
        event_day_rule = alt.Chart(pd.DataFrame({"relative_day": [0]})).mark_rule(
            color="#d62728",
            strokeDash=[5, 4],
            strokeWidth=1.5,
        ).encode(x="relative_day:Q")
        zero_return_rule = alt.Chart(pd.DataFrame({"average_CAR_display": [0]})).mark_rule(
            color="#7f7f7f",
            strokeDash=[2, 3],
            strokeWidth=1,
        ).encode(y="average_CAR_display:Q")
        car_window_bands = pd.DataFrame(
            [
                {"window": "CAR_-3_-1", "start": -3, "end": -1},
                {"window": "CAR_-1_3", "start": -1, "end": 3},
                {"window": "CAR_1_60", "start": 1, "end": 60},
            ]
        )
        if selected_shading_window != "All windows":
            car_window_bands = car_window_bands[car_window_bands["window"] == selected_shading_window]
        window_band_chart = (
            alt.Chart(car_window_bands)
            .mark_rect(opacity=0.12)
            .encode(
                x=alt.X("start:Q", title="Relative trading day"),
                x2="end:Q",
                color=alt.Color(
                    "window:N",
                    title="CAR window",
                    scale=alt.Scale(
                        domain=["CAR_-3_-1", "CAR_-1_3", "CAR_1_60"],
                        range=["#4e79a7", "#f28e2b", "#59a14f"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("window:N", title="CAR window"),
                    alt.Tooltip("start:Q", title="Start"),
                    alt.Tooltip("end:Q", title="End"),
                ],
            )
        )
        all_events_chart = alt.layer(
            window_band_chart,
            all_events_line,
            event_day_rule,
            zero_return_rule,
        ).resolve_scale(color="independent")
        st.altair_chart(all_events_chart, use_container_width=True)

        st.subheader("By Topic")
        if filtered_topic_car.empty:
            st.info("No topic rows match the selected filters.")
        else:
            by_topic_line = (
                alt.Chart(filtered_topic_car)
                .mark_line(strokeWidth=2.2)
                .encode(
                    x=alt.X("relative_day:Q", title="Relative trading day"),
                    y=alt.Y("average_CAR_display:Q", title="Average CAR", axis=alt.Axis(format="%")),
                    color=alt.Color("topic:N", title="Topic"),
                    tooltip=[
                        alt.Tooltip("topic:N", title="Topic"),
                        alt.Tooltip("relative_day:Q", title="Relative day"),
                        alt.Tooltip("average_CAR_display:Q", title="Average CAR", format=".2%"),
                        alt.Tooltip("average_abnormal_return:Q", title="Average abnormal return", format=".2%"),
                        alt.Tooltip("n_events:Q", title="Events"),
                    ],
                )
            )
            by_topic_chart = alt.layer(
                window_band_chart,
                by_topic_line,
                event_day_rule,
                zero_return_rule,
            ).resolve_scale(color="independent")
            st.altair_chart(by_topic_chart, use_container_width=True)

        with st.expander("Topic CAR Code Logic"):
            st.code(
                '''event_time_labeled = event_time.merge(
    events[["event_id", "topic", "topic_detail", "severity_bucket", "industry"]],
    on="event_id",
    how="left",
)

topic_average_car = event_time_labeled.groupby(["topic", "relative_day"], as_index=False).agg(
    average_abnormal_return=("abnormal_return", "mean"),
    n_events=("event_id", "nunique"),
).sort_values(["topic", "relative_day"])
topic_average_car["average_CAR"] = topic_average_car.groupby("topic")[
    "average_abnormal_return"
].cumsum()''',
                language="python",
            )

        with st.expander("Static Figure Snapshots"):
            cols = st.columns(2)
            if average_plot_path.exists():
                cols[0].image(str(average_plot_path), use_container_width=True)
            if topic_plot_path.exists():
                cols[1].image(str(topic_plot_path), use_container_width=True)

        st.subheader("Topic CAR Conclusions")
        st.markdown(
            f"""
- **Market reaction exists, but it is mixed.** Immediate CAR ranges from `{immediate_low['topic']}`
({format_percent(immediate_low['mean_CAR_immediate'])}) to `{immediate_high['topic']}`
({format_percent(immediate_high['mean_CAR_immediate'])}). This gives mixed support for H1, not a clean negative-reaction story.
- **Topic heterogeneity is visible.** Different topics produce different CAR patterns, which supports the descriptive part of H3.
- **Post-event drift exists for some topics.** Drift ranges from `{drift_low['topic']}`
({format_percent(drift_low['mean_CAR_drift'])}) to `{drift_high['topic']}`
({format_percent(drift_high['mean_CAR_drift'])}). This gives partial support for H2.
- **Overall conclusion:** Topic CAR suggests SEC comment letters contain market-relevant signals, but the direction and strength depend on content.
"""
        )

    with tabs[1]:
        st.header("Regression Playground")
        st.write(
            "This section presents the official 12-model regression suite: "
            "3 CAR windows by 4 control specifications."
        )
        st.info(
            "Regression models use `0 + C(topic)`, so all six topic groups are shown directly. "
            "Each topic coefficient is a topic-specific conditional CAR level, not a difference from an omitted baseline topic."
        )

        design_summary = pd.DataFrame(
            [
                {"Dimension": "Outcomes", "Choices": "CAR[-3,-1], CAR[-1,+3], CAR[+1,+60]"},
                {"Dimension": "Main variables", "Choices": "Topic fixed effects and severity_score"},
                {"Dimension": "Controls", "Choices": "None, pre-event volatility, industry, both controls"},
                {"Dimension": "Topic coding", "Choices": "No-intercept topic fixed effects: 0 + C(topic)"},
                {"Dimension": "Standard errors", "Choices": "HC1 robust standard errors"},
            ]
        )
        st.subheader("Regression Design")
        st.dataframe(design_summary, use_container_width=True, hide_index=True)

        st.subheader("Main Regression Summary")
        st.caption(
            "Cells show coefficient with HC1 robust standard error in parentheses. "
            "*, **, *** denote p < 0.10, 0.05, and 0.01. Coefficients are in return units, so 0.0100 means 1.00% CAR."
        )
        outcome_tabs = st.tabs(list(OUTCOME_LABELS))
        for outcome_tab, outcome_label in zip(outcome_tabs, OUTCOME_LABELS):
            outcome = OUTCOME_LABELS[outcome_label]
            with outcome_tab:
                st.write(outcome_description(outcome))
                st.dataframe(
                    build_outcome_regression_table(regression_coefficients, outcome_label),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Model diagnostics for the four specifications in this CAR window.")
                st.dataframe(
                    build_outcome_diagnostics_table(regression_model_summary, outcome_label),
                    use_container_width=True,
                    hide_index=True,
                )

        st.subheader("Interpretation Notes")
        st.markdown(
            """
- Topic coefficients compare topic-specific conditional CAR levels across SEC comment-letter types.
- `severity_score` measures whether more severe letters within topic groups are associated with higher or lower CAR.
- The four specifications show whether the topic/severity patterns are stable after adding pre-event volatility and industry controls.
- The immediate window, `CAR[-1,+3]`, is the main short-window market-reaction test; `CAR[-3,-1]` checks pre-event movement and `CAR[+1,+60]` checks post-event drift.
"""
        )

        with st.expander("Full 12-Model Coefficient Table"):
            coefficient_cols = [
                "model_id",
                "outcome_label",
                "spec_label",
                "term_label",
                "term_group",
                "coefficient",
                "std_error",
                "t_stat",
                "p_value",
                "coefficient_display",
                "formula",
                "n_obs",
            ]
            st.dataframe(regression_coefficients[coefficient_cols], use_container_width=True, hide_index=True)

        with st.expander("Model Formulas and Diagnostics"):
            model_cols = [
                "model_id",
                "outcome_label",
                "spec_label",
                "n_obs",
                "r_squared",
                "adj_r_squared",
                "include_pre_event_volatility",
                "include_industry",
                "formula",
            ]
            st.dataframe(regression_model_summary[model_cols], use_container_width=True, hide_index=True)

        with st.expander("Regression-Ready Analysis Dataset"):
            st.write(
                "This is the merged event-level table used for the regression suite: "
                "text features, topic classifier output, severity score, market-adjusted CAR windows, "
                "and the pre-event volatility control all sit on the same event row."
            )
            filter_cols = st.columns([1, 1, 1, 1])
            with filter_cols[0]:
                status_options = sorted(reg_df["regression_inclusion_status"].dropna().unique().tolist())
                selected_status = st.multiselect(
                    "Inclusion status",
                    status_options,
                    default=["included"] if "included" in status_options else status_options,
                    key="regression_ready_status",
                )
            with filter_cols[1]:
                dataset_topic_options = sorted(reg_df["topic"].dropna().unique().tolist())
                selected_dataset_topics = st.multiselect(
                    "Dataset topics",
                    dataset_topic_options,
                    default=dataset_topic_options,
                    key="regression_ready_topics",
                )
            with filter_cols[2]:
                industry_options = sorted(reg_df["industry"].dropna().unique().tolist())
                selected_industries = st.multiselect(
                    "Industries",
                    industry_options,
                    default=industry_options,
                    key="regression_ready_industries",
                )
            with filter_cols[3]:
                ticker_query = st.text_input("Ticker search", value="", key="regression_ready_ticker")

            analysis_df = reg_df.copy()
            if selected_status:
                analysis_df = analysis_df[analysis_df["regression_inclusion_status"].isin(selected_status)]
            if selected_dataset_topics:
                analysis_df = analysis_df[analysis_df["topic"].isin(selected_dataset_topics)]
            if selected_industries:
                analysis_df = analysis_df[analysis_df["industry"].isin(selected_industries)]
            if ticker_query.strip():
                analysis_df = analysis_df[
                    analysis_df["ticker"].astype(str).str.upper().str.contains(ticker_query.strip().upper(), na=False)
                ]

            analysis_cols = [
                "event_id",
                "ticker",
                "firm_name",
                "event_date",
                "industry",
                "topic",
                "topic_detail",
                "topic_classifier_version",
                "severity_score",
                "severity_bucket",
                "CAR_-3_-1",
                "CAR_-1_3",
                "CAR_1_60",
                "pre_event_volatility",
                "regression_inclusion_status",
            ]
            st.caption(
                f"Showing {len(analysis_df):,} rows. CAR columns use stock return minus SPY return; "
                "pre-event volatility is event-specific over relative trading days [-150,-31]."
            )
            st.dataframe(analysis_df[analysis_cols], use_container_width=True, hide_index=True)

        with st.expander("Why Regression Matters"):
            st.markdown(
                """
Event study asks whether the market reacts around SEC comment-letter events.
Regression asks whether that market reaction can be explained by the content of the comment letters.

Event study is useful for documenting whether average CAR is positive or negative,
whether different topics have different average CAR paths, and whether reactions appear around
the event date or continue as post-event drift.

Regression adds the cross-sectional test. It asks whether some topics are associated with
stronger market reactions, whether higher `severity_score` corresponds to larger CAR,
and whether these relationships remain after controlling for industry and pre-event volatility.

In this project, the regression analysis tests whether SEC comment-letter content features,
especially `topic` and `severity_score`, explain variation in CAR across events.

| Part | Research role |
|---|---|
| Event study | Establish whether market reaction exists |
| Topic CAR | Descriptive comparison across topics |
| Regression | Formal test of whether topic and severity explain CAR differences |

Presentation interpretation: the event study documents market reaction around SEC comment-letter
events, while the regression analysis tests whether the magnitude and direction of that reaction
are systematically related to letter content, after accounting for stock volatility and industry differences.
"""
            )

        st.subheader("Regression Conclusions")
        st.markdown(
            f"""
- **Regression evidence is weaker than Topic CAR.** Best R-squared is {max_r_squared:.3f}; best adjusted R-squared is {max_adj_r_squared:.3f}.
- **Severity is not supported.** `severity_score` is significant in {severity_sig_count}/12 models, so H4 is not supported in this baseline.
- **Topic has partial evidence.** Topic coefficients are significant {topic_sig_count} times across the 12-model suite, giving limited exploratory support for H3.
- **Overall conclusion:** Regression does not fail the project; it shows that topic/severity have limited explanatory power for cross-sectional CAR in the current 100-company sample.
"""
        )
