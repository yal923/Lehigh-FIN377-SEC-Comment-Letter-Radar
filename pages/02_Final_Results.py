from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"


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


def control_description(include_industry: bool, include_pre_volatility: bool) -> str:
    parts = []
    if include_industry:
        parts.append("industry fixed effects compare events within broad sector groups")
    if include_pre_volatility:
        parts.append("pre-event volatility controls for noisier stocks having larger CAR mechanically")
    if not parts:
        return "No controls: this is the simplest topic/severity relationship."
    return "; ".join(parts) + "."


def run_playground_regression(
    df: pd.DataFrame,
    outcome: str,
    include_industry: bool,
    include_pre_volatility: bool,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    terms = ["severity_score", "C(topic)"]
    required = [outcome, "severity_score", "topic"]
    if include_industry:
        terms.append("C(industry)")
        required.append("industry")
    if include_pre_volatility:
        terms.append("pre_event_volatility")
        required.append("pre_event_volatility")

    regression_input = df[df["regression_inclusion_status"] == "included"].copy()
    regression_input = regression_input.dropna(subset=required)
    formula = f'Q("{outcome}") ~ ' + " + ".join(terms)

    if len(regression_input) < 3 or regression_input["topic"].nunique() < 2:
        coef = pd.DataFrame(
            [
                {
                    "term": "not_run",
                    "coefficient": None,
                    "std_error": None,
                    "t_stat": None,
                    "p_value": None,
                    "formula": formula,
                    "n_obs": len(regression_input),
                }
            ]
        )
        return coef, formula, regression_input

    model = smf.ols(formula, data=regression_input).fit(cov_type="HC1")
    coef = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "formula": formula,
            "n_obs": int(model.nobs),
        }
    )
    return coef, formula, regression_input


topic_summary_path = TABLES_DIR / "topic_car_summary.csv"
topic_average_car_path = TABLES_DIR / "topic_average_car.csv"
average_car_path = TABLES_DIR / "average_car.csv"
event_summary_path = TABLES_DIR / "event_summary.csv"
regression_dataset_path = PROCESSED_DIR / "regression_dataset.parquet"
topic_plot_path = FIGURES_DIR / "topic_average_car.png"
average_plot_path = FIGURES_DIR / "average_car_plot.png"

required = [topic_summary_path, topic_average_car_path, average_car_path, regression_dataset_path]
if not all(path.exists() for path in required):
    st.warning("Official result artifacts are not available yet. Run `./.venv/bin/python run.py --step event_study`.")
else:
    topic_summary = read_csv(topic_summary_path)
    topic_average_car = read_csv(topic_average_car_path)
    average_car = read_csv(average_car_path)
    reg_df = read_regression_dataset()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Events", reg_df["event_id"].nunique())
    metric_cols[1].metric("Regression included", int((reg_df["regression_inclusion_status"] == "included").sum()))
    metric_cols[2].metric("Topic groups", topic_summary["topic"].nunique())
    metric_cols[3].metric("Industries", reg_df["industry"].nunique())

    tabs = st.tabs(["Topic CAR", "Regression Playground", "Result Tables", "Static Figures"])

    with tabs[0]:
        st.header("Topic CAR")
        st.write("This view compares average cumulative abnormal returns across topic groups.")

        available_topics = sorted(topic_average_car["topic"].dropna().unique().tolist())
        default_topics = available_topics
        selected_topics = st.multiselect("Topics", available_topics, default=default_topics)
        day_min, day_max = int(topic_average_car["relative_day"].min()), int(topic_average_car["relative_day"].max())
        selected_window = st.slider("Relative trading-day window", day_min, day_max, (day_min, day_max))

        filtered_topic_car = topic_average_car[
            topic_average_car["topic"].isin(selected_topics)
            & topic_average_car["relative_day"].between(selected_window[0], selected_window[1])
        ]
        filtered_average_car = average_car[average_car["relative_day"].between(selected_window[0], selected_window[1])]

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.subheader("All Events")
            st.line_chart(filtered_average_car.set_index("relative_day")["average_CAR"])
        with chart_cols[1]:
            st.subheader("By Topic")
            if filtered_topic_car.empty:
                st.info("Select at least one topic.")
            else:
                chart_data = filtered_topic_car.pivot(index="relative_day", columns="topic", values="average_CAR")
                st.line_chart(chart_data)

        st.subheader("Topic CAR Summary")
        visible_summary = topic_summary[topic_summary["topic"].isin(selected_topics)] if selected_topics else topic_summary.iloc[0:0]
        st.dataframe(visible_summary, use_container_width=True)

        with st.expander("Underlying topic CAR rows"):
            st.dataframe(filtered_topic_car, use_container_width=True)

    with tabs[1]:
        st.header("Regression Playground")
        st.write("Main variables stay fixed: `topic` and `severity_score`. Use the controls to test robustness.")

        outcome_labels = {
            "Immediate reaction: CAR[-1,+3]": "CAR_-1_3",
            "Pre-event movement: CAR[-3,-1]": "CAR_-3_-1",
            "Post-event drift: CAR[+1,+60]": "CAR_1_60",
        }
        left, right = st.columns([1, 2])
        with left:
            selected_outcome_label = st.selectbox("Outcome", list(outcome_labels))
            selected_outcome = outcome_labels[selected_outcome_label]
            include_industry = st.checkbox("Industry controls", value=False)
            include_pre_volatility = st.checkbox("Pre-event volatility control", value=True)
        with right:
            st.subheader("Specification")
            st.write(outcome_description(selected_outcome))
            st.write(control_description(include_industry, include_pre_volatility))

        coef, formula, regression_input = run_playground_regression(
            reg_df,
            selected_outcome,
            include_industry,
            include_pre_volatility,
        )

        spec_cols = st.columns(3)
        spec_cols[0].metric("Observations", len(regression_input))
        spec_cols[1].metric("Topics", regression_input["topic"].nunique() if not regression_input.empty else 0)
        spec_cols[2].metric("Industries", regression_input["industry"].nunique() if not regression_input.empty else 0)
        st.code(formula, language="text")

        st.subheader("Coefficients")
        st.dataframe(coef, use_container_width=True)

        plot_coef = coef[
            coef["term"].ne("Intercept")
            & coef["coefficient"].notna()
            & ~coef["term"].str.startswith("C(industry)", na=False)
        ][["term", "coefficient"]]
        if not plot_coef.empty:
            st.bar_chart(plot_coef.set_index("term"))

        with st.expander("Regression input rows"):
            preview_cols = [
                "event_id",
                "ticker",
                "event_date",
                "topic",
                "industry",
                "severity_score",
                selected_outcome,
                "pre_event_volatility",
            ]
            st.dataframe(regression_input[preview_cols], use_container_width=True)

    with tabs[2]:
        st.header("Result Tables")
        st.subheader("Topic CAR Summary")
        st.dataframe(topic_summary, use_container_width=True)

        st.subheader("Average CAR")
        st.dataframe(average_car, use_container_width=True)

        if event_summary_path.exists():
            st.subheader("Event Summary")
            st.dataframe(read_csv(event_summary_path), use_container_width=True)

    with tabs[3]:
        st.header("Static Figures")
        cols = st.columns(2)
        if average_plot_path.exists():
            cols[0].image(str(average_plot_path), use_container_width=True)
        if topic_plot_path.exists():
            cols[1].image(str(topic_plot_path), use_container_width=True)
