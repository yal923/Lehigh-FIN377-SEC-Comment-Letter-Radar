from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sec_letter_project_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StepResult:
    step_id: str
    step_name: str
    status: str
    source: str
    output_files: list[str]
    metrics: dict[str, Any]
    notes: list[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet_or_csv(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing input table: {parquet_path}")


def nearest_future_trading_day(trading_dates: pd.Series, target_date: pd.Timestamp) -> pd.Timestamp | None:
    future_dates = trading_dates[trading_dates >= target_date]
    if future_dates.empty:
        return None
    return future_dates.min()


def build_event_time(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    event_window_start: int,
    event_window_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["return"] = pd.to_numeric(prices["return"], errors="coerce")
    market = (
        prices[prices["ticker"] == benchmark_ticker][["date", "return"]]
        .rename(columns={"return": "market_return"})
        .dropna(subset=["date"])
    )
    stock_prices = prices[prices["ticker"] != benchmark_ticker].merge(market, on="date", how="left")
    stock_prices["abnormal_return"] = stock_prices["return"] - stock_prices["market_return"]

    output_rows: list[pd.DataFrame] = []
    qa_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        firm_data = stock_prices[stock_prices["ticker"] == event.ticker].sort_values("date").reset_index(drop=True)
        if firm_data.empty:
            qa_rows.append({"event_id": event.event_id, "event_time_status": "missing_ticker_market_data", "n_event_time_rows": 0})
            continue

        event_date = pd.Timestamp(event.event_date)
        actual_event_day = nearest_future_trading_day(firm_data["date"], event_date)
        if actual_event_day is None:
            qa_rows.append({"event_id": event.event_id, "event_time_status": "no_future_trading_day", "n_event_time_rows": 0})
            continue

        event_idx_list = firm_data.index[firm_data["date"] == actual_event_day].tolist()
        if not event_idx_list:
            qa_rows.append({"event_id": event.event_id, "event_time_status": "event_day_index_missing", "n_event_time_rows": 0})
            continue

        event_idx = event_idx_list[0]
        start_idx = max(0, event_idx + event_window_start)
        end_idx = min(len(firm_data) - 1, event_idx + event_window_end)
        window = firm_data.iloc[start_idx : end_idx + 1].copy()
        window["event_id"] = event.event_id
        window["event_date"] = event.event_date
        window["actual_event_trading_day"] = actual_event_day.date().isoformat()
        window["relative_day"] = window.index - event_idx
        window = window[(window["relative_day"] >= event_window_start) & (window["relative_day"] <= event_window_end)].copy()
        window["CAR"] = window["abnormal_return"].fillna(0).cumsum()
        window["volatility_proxy"] = window["return"].rolling(5, min_periods=2).std()
        output_rows.append(
            window[
                [
                    "event_id",
                    "ticker",
                    "event_date",
                    "actual_event_trading_day",
                    "date",
                    "relative_day",
                    "return",
                    "market_return",
                    "abnormal_return",
                    "CAR",
                    "volume",
                    "volatility_proxy",
                ]
            ]
        )
        qa_rows.append({"event_id": event.event_id, "event_time_status": "ok", "n_event_time_rows": int(len(window))})

    event_time = pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame()
    qa = pd.DataFrame(qa_rows)
    return event_time, qa


def car_window(event_time: pd.DataFrame, start: int, end: int) -> pd.Series:
    temp = event_time[(event_time["relative_day"] >= start) & (event_time["relative_day"] <= end)]
    return temp.groupby("event_id")["abnormal_return"].sum().rename(f"CAR_{start}_{end}")


def build_regression_dataset(events: pd.DataFrame, event_time: pd.DataFrame) -> pd.DataFrame:
    cars = pd.concat(
        [
            car_window(event_time, -3, -1),
            car_window(event_time, -1, 3),
            car_window(event_time, 1, 60),
        ],
        axis=1,
    ).reset_index()
    pre_vol = (
        event_time[(event_time["relative_day"] >= -20) & (event_time["relative_day"] <= -2)]
        .groupby("event_id")["return"]
        .std()
        .rename("pre_event_volatility")
        .reset_index()
    )
    reg = events.merge(cars, on="event_id", how="left").merge(pre_vol, on="event_id", how="left")
    required = ["CAR_-1_3", "severity_score", "pre_event_volatility", "topic"]
    reg["regression_inclusion_status"] = "included"
    reg.loc[~reg[required].notna().all(axis=1), "regression_inclusion_status"] = "excluded_missing_required_field"
    return reg


def run_ols(regression_dataset: pd.DataFrame, outcome: str) -> pd.DataFrame:
    regression_input = regression_dataset[regression_dataset["regression_inclusion_status"] == "included"].copy()
    topic_n = regression_input["topic"].nunique(dropna=True)
    if len(regression_input) >= 8 and topic_n >= 2:
        formula = f'Q("{outcome}") ~ severity_score + pre_event_volatility + C(topic)'
    else:
        formula = f'Q("{outcome}") ~ severity_score + pre_event_volatility'

    if len(regression_input) < 3:
        return pd.DataFrame(
            [
                {
                    "window": outcome,
                    "term": "not_run",
                    "formula": formula,
                    "n_obs": len(regression_input),
                    "reason": "fewer_than_3_observations",
                }
            ]
        )

    model = smf.ols(formula, data=regression_input).fit(cov_type="HC1")
    return pd.DataFrame(
        {
            "window": outcome,
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "formula": formula,
            "n_obs": int(model.nobs),
        }
    )


def write_outputs(
    root: Path,
    event_time: pd.DataFrame,
    event_time_qa: pd.DataFrame,
    regression_dataset: pd.DataFrame,
    events: pd.DataFrame,
) -> StepResult:
    processed_dir = root / "data" / "processed"
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    event_time_csv = processed_dir / "event_time.csv"
    event_time_parquet = processed_dir / "event_time.parquet"
    event_time_qa_path = processed_dir / "event_time_qa.csv"
    regression_csv = processed_dir / "regression_dataset.csv"
    regression_parquet = processed_dir / "regression_dataset.parquet"
    event_time_preview_path = workflow_dir / "07_event_time_preview.csv"
    regression_preview_path = workflow_dir / "07_regression_dataset_preview.csv"
    event_time_qa_summary_path = tables_dir / "event_time_qa_summary.csv"
    average_car_path = tables_dir / "average_car.csv"
    topic_average_car_path = tables_dir / "topic_average_car.csv"
    topic_car_summary_path = tables_dir / "topic_car_summary.csv"
    event_summary_path = tables_dir / "event_summary.csv"
    baseline_regression_path = tables_dir / "baseline_regression.csv"
    regression_pre_path = tables_dir / "regression_CAR_-3_-1.csv"
    regression_drift_path = tables_dir / "regression_CAR_1_60.csv"
    average_car_plot_path = figures_dir / "average_car_plot.png"
    topic_average_car_plot_path = figures_dir / "topic_average_car.png"
    summary_path = workflow_dir / "07_event_study_regression_summary.json"

    event_time.to_csv(event_time_csv, index=False)
    event_time.to_parquet(event_time_parquet, index=False)
    event_time_qa.to_csv(event_time_qa_path, index=False)
    regression_dataset.to_csv(regression_csv, index=False)
    regression_dataset.to_parquet(regression_parquet, index=False)
    event_time.head(150).to_csv(event_time_preview_path, index=False)

    regression_preview_cols = [
        "event_id",
        "ticker",
        "event_date",
        "topic",
        "severity_score",
        "CAR_-3_-1",
        "CAR_-1_3",
        "CAR_1_60",
        "pre_event_volatility",
        "regression_inclusion_status",
    ]
    regression_dataset[regression_preview_cols].head(150).to_csv(regression_preview_path, index=False)

    event_time_qa_summary = (
        event_time_qa.groupby("event_time_status", dropna=False)
        .agg(event_count=("event_id", "nunique"), total_rows=("n_event_time_rows", "sum"))
        .reset_index()
        .sort_values("event_count", ascending=False)
    )
    event_time_qa_summary.to_csv(event_time_qa_summary_path, index=False)

    event_time_labeled = event_time.merge(
        events[["event_id", "topic", "topic_detail", "severity_bucket", "industry"]],
        on="event_id",
        how="left",
    )
    average_car = event_time_labeled.groupby("relative_day", as_index=False).agg(
        average_CAR=("CAR", "mean"),
        average_abnormal_return=("abnormal_return", "mean"),
        n_events=("event_id", "nunique"),
    )
    topic_average_car = event_time_labeled.groupby(["topic", "relative_day"], as_index=False).agg(
        average_CAR=("CAR", "mean"),
        average_abnormal_return=("abnormal_return", "mean"),
        n_events=("event_id", "nunique"),
    )
    topic_car_summary = regression_dataset.groupby("topic", as_index=False).agg(
        n_events=("event_id", "nunique"),
        mean_CAR_pre=("CAR_-3_-1", "mean"),
        mean_CAR_immediate=("CAR_-1_3", "mean"),
        mean_CAR_drift=("CAR_1_60", "mean"),
        mean_severity=("severity_score", "mean"),
    )
    average_car.to_csv(average_car_path, index=False)
    topic_average_car.to_csv(topic_average_car_path, index=False)
    topic_car_summary.to_csv(topic_car_summary_path, index=False)
    regression_dataset[
        [
            "event_id",
            "ticker",
            "event_date",
            "topic",
            "topic_detail",
            "severity_score",
            "CAR_-3_-1",
            "CAR_-1_3",
            "CAR_1_60",
            "pre_event_volatility",
            "regression_inclusion_status",
        ]
    ].to_csv(event_summary_path, index=False)

    baseline_coef = run_ols(regression_dataset, "CAR_-1_3")
    pre_coef = run_ols(regression_dataset, "CAR_-3_-1")
    drift_coef = run_ols(regression_dataset, "CAR_1_60")
    baseline_coef.to_csv(baseline_regression_path, index=False)
    pre_coef.to_csv(regression_pre_path, index=False)
    drift_coef.to_csv(regression_drift_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(average_car["relative_day"], average_car["average_CAR"])
    plt.axvline(0, linestyle="--", color="black", linewidth=1)
    plt.title("Average CAR Around SEC Comment-Letter Events")
    plt.xlabel("Relative trading day")
    plt.ylabel("Average CAR")
    plt.tight_layout()
    plt.savefig(average_car_plot_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for topic, group in topic_average_car.groupby("topic"):
        plt.plot(group["relative_day"], group["average_CAR"], label=topic)
    plt.axvline(0, linestyle="--", color="black", linewidth=1)
    plt.title("Topic-Average CAR Around SEC Comment-Letter Events")
    plt.xlabel("Relative trading day")
    plt.ylabel("Average CAR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(topic_average_car_plot_path, dpi=300)
    plt.close()

    metrics = {
        "event_time_row_count": int(len(event_time)),
        "event_time_event_count": int(event_time["event_id"].nunique()) if not event_time.empty else 0,
        "event_time_ok_count": int((event_time_qa["event_time_status"] == "ok").sum()) if not event_time_qa.empty else 0,
        "regression_row_count": int(len(regression_dataset)),
        "regression_included_count": int((regression_dataset["regression_inclusion_status"] == "included").sum()),
        "topic_count": int(regression_dataset["topic"].nunique(dropna=True)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    notes = [
        "Constructed event-time rows using the next available trading day on or after each SEC event date.",
        "Abnormal return is stock return minus benchmark return.",
        "Regression tables are generated for CAR[-3,-1], CAR[-1,+3], and CAR[+1,+60].",
    ]
    result = StepResult(
        step_id="07_event_study_regression",
        step_name="Event Study and Regression",
        status="complete",
        source="data/processed/event_level and market_data",
        output_files=[
            str(event_time_csv.relative_to(root)),
            str(event_time_parquet.relative_to(root)),
            str(event_time_qa_path.relative_to(root)),
            str(regression_csv.relative_to(root)),
            str(regression_parquet.relative_to(root)),
            str(event_time_preview_path.relative_to(root)),
            str(regression_preview_path.relative_to(root)),
            str(event_time_qa_summary_path.relative_to(root)),
            str(average_car_path.relative_to(root)),
            str(topic_average_car_path.relative_to(root)),
            str(topic_car_summary_path.relative_to(root)),
            str(event_summary_path.relative_to(root)),
            str(baseline_regression_path.relative_to(root)),
            str(regression_pre_path.relative_to(root)),
            str(regression_drift_path.relative_to(root)),
            str(average_car_plot_path.relative_to(root)),
            str(topic_average_car_plot_path.relative_to(root)),
            str(summary_path.relative_to(root)),
        ],
        metrics=metrics,
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def build_event_study_regression(root: Path | None = None) -> StepResult:
    root = root or project_root()
    config = load_json(root / "configs" / "sample_100_10y.json")
    events = read_parquet_or_csv(root / "data" / "processed" / "event_level.parquet", root / "data" / "processed" / "event_level.csv")
    prices = read_parquet_or_csv(root / "data" / "processed" / "market_data.parquet", root / "data" / "processed" / "market_data.csv")
    event_time, event_time_qa = build_event_time(
        events=events,
        prices=prices,
        benchmark_ticker=config.get("benchmark_ticker", "SPY"),
        event_window_start=int(config.get("event_window_start", -3)),
        event_window_end=int(config.get("event_window_end", 60)),
    )
    regression_dataset = build_regression_dataset(events, event_time)
    return write_outputs(root, event_time, event_time_qa, regression_dataset, events)


if __name__ == "__main__":
    result = build_event_study_regression()
    print(json.dumps(asdict(result), indent=2))
