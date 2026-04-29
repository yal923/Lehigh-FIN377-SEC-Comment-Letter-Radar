from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


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


def read_event_base(root: Path) -> pd.DataFrame:
    processed_dir = root / "data" / "processed"
    parquet_path = processed_dir / "event_level_base.parquet"
    csv_path = processed_dir / "event_level_base.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing event-level base dataset: {parquet_path}")


def count_terms(text: str, terms: list[str]) -> int:
    text_lower = str(text).lower()
    return sum(text_lower.count(term.lower()) for term in terms)


def count_regex_patterns(text: str, patterns: list[str]) -> int:
    text = str(text)
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def topic_scores(text: str, config: dict[str, Any]) -> dict[str, int]:
    if config.get("topic_patterns"):
        return {
            topic: count_regex_patterns(text, patterns)
            for topic, patterns in config["topic_patterns"].items()
        }
    return {
        topic: count_terms(text, terms)
        for topic, terms in config.get("topic_keywords", {}).items()
        if topic != "other"
    }


def routine_disclosure_score(text: str, config: dict[str, Any]) -> int:
    return count_regex_patterns(text, config.get("routine_disclosure_patterns", []))


def classify_topic_detail_from_scores(scores: dict[str, int], routine_score: int, config: dict[str, Any]) -> str:
    positive_scores = {topic: score for topic, score in scores.items() if score > 0}
    if positive_scores:
        priority = {topic: idx for idx, topic in enumerate(config.get("topic_priority", []))}
        return sorted(
            positive_scores,
            key=lambda topic: (-positive_scores[topic], priority.get(topic, 999), topic),
        )[0]
    if routine_score > 0:
        return "routine_disclosure"
    return "other"


def map_topic_group(topic_detail: str, config: dict[str, Any]) -> str:
    return config.get("topic_group_mapping", {}).get(topic_detail, topic_detail)


def zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_text_features(event_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = event_df.copy()
    df["letter_text"] = df.get("staff_letter_text", "").fillna("")
    df["all_thread_text"] = df.get("all_thread_text", "").fillna("")
    df["letter_word_count"] = pd.to_numeric(df.get("staff_letter_word_count", 0), errors="coerce").fillna(0).astype(int)

    scores = df["letter_text"].apply(lambda text: topic_scores(text, config))
    routine_scores = df["letter_text"].apply(lambda text: routine_disclosure_score(text, config))
    df["topic_score_json"] = scores.apply(json.dumps)
    df["routine_disclosure_score"] = routine_scores
    df["topic_classifier_version"] = config.get("topic_classifier_version", "v1_keyword_count")
    df["topic_detail"] = [
        classify_topic_detail_from_scores(score, routine_score, config)
        for score, routine_score in zip(scores, routine_scores)
    ]
    df["topic"] = df["topic_detail"].apply(lambda topic_detail: map_topic_group(topic_detail, config))
    df["intensity_term_count"] = df["letter_text"].apply(lambda text: count_terms(text, config.get("intensity_terms", [])))
    df["amendment_indicator"] = df["all_thread_text"].str.contains(
        r"\bamend|amended|amendment\b",
        case=False,
        regex=True,
        na=False,
    ).astype(int)

    df["z_letter_word_count"] = zscore(df["letter_word_count"])
    df["z_intensity_term_count"] = zscore(df["intensity_term_count"])
    df["z_thread_round_count"] = zscore(df["thread_round_count"])
    df["severity_score"] = (
        df["z_letter_word_count"]
        + df["z_intensity_term_count"]
        + df["z_thread_round_count"]
        + df["amendment_indicator"]
    )

    if df["severity_score"].nunique(dropna=True) >= 3:
        df["severity_bucket"] = pd.qcut(
            df["severity_score"],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
        df["severity_bucket"] = df["severity_bucket"].astype(str).replace("nan", "medium")
    else:
        df["severity_bucket"] = "medium"
    return df


def write_outputs(root: Path, event_df: pd.DataFrame, config: dict[str, Any]) -> StepResult:
    processed_dir = root / "data" / "processed"
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    event_csv = processed_dir / "event_level.csv"
    event_parquet = processed_dir / "event_level.parquet"
    preview_path = workflow_dir / "05_text_features_preview.csv"
    topic_summary_path = tables_dir / "topic_summary.csv"
    topic_detail_summary_path = tables_dir / "topic_detail_summary.csv"
    severity_summary_path = tables_dir / "severity_bucket_summary.csv"
    summary_path = workflow_dir / "05_text_features_summary.json"

    event_df.to_csv(event_csv, index=False)
    event_df.to_parquet(event_parquet, index=False)

    preview_cols = [
        "event_id",
        "event_date",
        "ticker",
        "industry",
        "topic",
        "topic_detail",
        "severity_score",
        "severity_bucket",
        "letter_word_count",
        "intensity_term_count",
        "thread_round_count",
        "amendment_indicator",
    ]
    event_df[preview_cols].head(100).to_csv(preview_path, index=False)

    topic_summary = (
        event_df.groupby("topic", dropna=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_severity=("severity_score", "mean"),
            median_severity=("severity_score", "median"),
            mean_letter_word_count=("letter_word_count", "mean"),
        )
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    topic_summary.to_csv(topic_summary_path, index=False)

    topic_detail_summary = (
        event_df.groupby(["topic", "topic_detail"], dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values(["topic", "n_events"], ascending=[True, False])
    )
    topic_detail_summary.to_csv(topic_detail_summary_path, index=False)

    severity_summary = (
        event_df.groupby("severity_bucket", dropna=False)
        .agg(
            n_events=("event_id", "nunique"),
            min_score=("severity_score", "min"),
            max_score=("severity_score", "max"),
            mean_score=("severity_score", "mean"),
        )
        .reset_index()
        .sort_values("severity_bucket")
    )
    severity_summary.to_csv(severity_summary_path, index=False)

    metrics = {
        "event_count": int(len(event_df)),
        "topic_count": int(event_df["topic"].nunique(dropna=True)),
        "topic_detail_count": int(event_df["topic_detail"].nunique(dropna=True)),
        "severity_bucket_count": int(event_df["severity_bucket"].nunique(dropna=True)),
        "mean_severity_score": round(float(event_df["severity_score"].mean()), 4),
        "classifier_version": config.get("topic_classifier_version"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    notes = [
        "Applied the validated V3 grouped rule-based regex topic classifier.",
        "Kept topic_detail as the audit label and topic as the grouped analysis label.",
        "Severity combines z-scored staff-letter length, intensity-term count, thread-round count, and amendment indicator.",
    ]
    result = StepResult(
        step_id="05_text_features",
        step_name="Text Features, Topic, and Severity",
        status="complete",
        source="data/processed/event_level_base and configs/text_feature_config.json",
        output_files=[
            str(event_csv.relative_to(root)),
            str(event_parquet.relative_to(root)),
            str(preview_path.relative_to(root)),
            str(topic_summary_path.relative_to(root)),
            str(topic_detail_summary_path.relative_to(root)),
            str(severity_summary_path.relative_to(root)),
            str(summary_path.relative_to(root)),
        ],
        metrics=metrics,
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def build_text_features(root: Path | None = None) -> StepResult:
    root = root or project_root()
    config = load_json(root / "configs" / "text_feature_config.json")
    event_base_df = read_event_base(root)
    event_df = add_text_features(event_base_df, config)
    return write_outputs(root, event_df, config)


if __name__ == "__main__":
    result = build_text_features()
    print(json.dumps(asdict(result), indent=2))
