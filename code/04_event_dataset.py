from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
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


def accession_no_dash(value: Any) -> str:
    return str(value).replace("-", "")


def read_table(csv_path: Path, parquet_path: Path | None = None) -> pd.DataFrame:
    if parquet_path is not None and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing required input table: {parquet_path or csv_path}")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = root / "data" / "processed"
    thread_df = read_table(
        processed_dir / "thread_level.csv",
        processed_dir / "thread_level.parquet",
    )
    filing_df = read_table(
        processed_dir / "thread_filing_manifest.csv",
        processed_dir / "thread_filing_manifest.parquet",
    )
    return thread_df, filing_df


def build_anchor_lookup(filing_df: pd.DataFrame) -> pd.DataFrame:
    if filing_df.empty:
        return pd.DataFrame(
            columns=[
                "thread_id",
                "event_anchor_accession",
                "main_staff_letter_url",
                "main_staff_letter_document_name",
                "main_staff_letter_extraction_status",
                "main_staff_letter_filing_date",
            ]
        )

    staff = filing_df[filing_df["normalized_form"] == "STAFF_LETTER"].copy()
    if staff.empty:
        return pd.DataFrame()

    staff = staff.sort_values(["ticker", "thread_id", "filing_date", "accession_number"])
    anchors = staff.groupby(["ticker", "thread_id"], as_index=False).first()
    anchors = anchors.rename(
        columns={
            "accession_number": "event_anchor_accession",
            "document_url": "main_staff_letter_url",
            "document_name": "main_staff_letter_document_name",
            "extraction_status": "main_staff_letter_extraction_status",
            "filing_date": "main_staff_letter_filing_date",
        }
    )
    return anchors[
        [
            "ticker",
            "thread_id",
            "event_anchor_accession",
            "main_staff_letter_url",
            "main_staff_letter_document_name",
            "main_staff_letter_extraction_status",
            "main_staff_letter_filing_date",
        ]
    ]


def reason_list(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if pd.isna(row.get("event_date")):
        reasons.append("missing_event_date")
    if pd.isna(row.get("ticker")) or str(row.get("ticker")).strip() == "":
        reasons.append("missing_ticker")
    if pd.isna(row.get("event_anchor_accession")) or str(row.get("event_anchor_accession")).strip() == "":
        reasons.append("missing_staff_letter_anchor")
    if pd.to_numeric(row.get("staff_letter_word_count"), errors="coerce") <= 0:
        reasons.append("empty_staff_letter_text")
    if bool(row.get("is_duplicate_event_id")):
        reasons.append("duplicate_event_id")
    return reasons


def build_event_level_raw(thread_df: pd.DataFrame, filing_df: pd.DataFrame) -> pd.DataFrame:
    if thread_df.empty:
        return pd.DataFrame()

    df = thread_df.copy()
    anchors = build_anchor_lookup(filing_df)
    merge_cols = ["ticker", "thread_id", "event_anchor_accession"]
    df = df.merge(anchors, on=merge_cols, how="left", suffixes=("", "_anchor"))

    df["event_date"] = pd.to_datetime(df["event_anchor_date"], errors="coerce")
    df["event_id"] = [
        f"{ticker}_{accession_no_dash(accession)}"
        for ticker, accession in zip(df["ticker"], df["event_anchor_accession"])
    ]
    df["is_duplicate_event_id"] = df["event_id"].duplicated(keep=False)
    df["event_construction_method"] = "first_staff_letter_in_thread"

    if "n_staff_letter" in df.columns:
        df["thread_round_count"] = pd.to_numeric(df["n_staff_letter"], errors="coerce").fillna(0).astype(int)
    else:
        df["thread_round_count"] = 0

    df["event_exclusion_reasons"] = df.apply(lambda row: ";".join(reason_list(row)), axis=1)
    df["event_inclusion_status"] = df["event_exclusion_reasons"].apply(lambda value: "included" if value == "" else "excluded")
    df["event_date"] = df["event_date"].dt.date.astype("string")

    ordered_cols = [
        "event_id",
        "event_inclusion_status",
        "event_exclusion_reasons",
        "event_construction_method",
        "event_date",
        "ticker",
        "thread_id",
        "firm_name",
        "cik",
        "industry",
        "event_anchor_accession",
        "main_staff_letter_url",
        "main_staff_letter_document_name",
        "main_staff_letter_extraction_status",
        "n_filings_in_thread",
        "n_staff_letter",
        "n_filer_response",
        "n_staff_action",
        "thread_round_count",
        "first_filing_date",
        "last_filing_date",
        "thread_duration_days",
        "forms_in_thread",
        "normalized_forms_in_thread",
        "accessions_json",
        "filing_dates_json",
        "document_names_json",
        "document_urls_json",
        "staff_letter_word_count",
        "response_word_count",
        "staff_action_word_count",
        "all_thread_word_count",
        "staff_letter_text",
        "response_text",
        "staff_action_text",
        "all_thread_text",
        "threading_status",
        "text_status",
    ]
    existing_cols = [col for col in ordered_cols if col in df.columns]
    return df[existing_cols].sort_values(["ticker", "event_date", "event_id"]).reset_index(drop=True)


def write_outputs(root: Path, event_raw_df: pd.DataFrame) -> StepResult:
    processed_dir = root / "data" / "processed"
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    event_raw_csv = processed_dir / "event_level_raw.csv"
    event_raw_parquet = processed_dir / "event_level_raw.parquet"
    event_base_csv = processed_dir / "event_level_base.csv"
    event_base_parquet = processed_dir / "event_level_base.parquet"
    preview_path = workflow_dir / "04_event_preview.csv"
    inclusion_summary_path = tables_dir / "event_inclusion_summary.csv"
    summary_path = workflow_dir / "04_event_dataset_summary.json"

    event_base_df = event_raw_df[event_raw_df["event_inclusion_status"] == "included"].copy()
    event_raw_df.to_csv(event_raw_csv, index=False)
    event_raw_df.to_parquet(event_raw_parquet, index=False)
    event_base_df.to_csv(event_base_csv, index=False)
    event_base_df.to_parquet(event_base_parquet, index=False)

    preview_cols = [
        "event_id",
        "event_inclusion_status",
        "event_date",
        "ticker",
        "firm_name",
        "industry",
        "thread_id",
        "n_filings_in_thread",
        "thread_round_count",
        "thread_duration_days",
        "staff_letter_word_count",
        "main_staff_letter_url",
    ]
    event_raw_df[preview_cols].head(100).to_csv(preview_path, index=False)

    inclusion_summary = (
        event_raw_df.groupby(["event_inclusion_status", "event_exclusion_reasons"], dropna=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["event_inclusion_status", "event_count"], ascending=[True, False])
    )
    inclusion_summary.to_csv(inclusion_summary_path, index=False)

    durations = pd.to_numeric(event_base_df["thread_duration_days"], errors="coerce") if not event_base_df.empty else pd.Series(dtype=float)
    metrics = {
        "raw_event_count": int(len(event_raw_df)),
        "included_event_count": int(len(event_base_df)),
        "excluded_event_count": int((event_raw_df["event_inclusion_status"] == "excluded").sum()) if not event_raw_df.empty else 0,
        "ticker_count": int(event_base_df["ticker"].nunique()) if not event_base_df.empty else 0,
        "industry_count": int(event_base_df["industry"].nunique()) if not event_base_df.empty else 0,
        "duplicate_event_id_count": int(event_raw_df["event_id"].duplicated(keep=False).sum()) if not event_raw_df.empty else 0,
        "mean_thread_duration_days": round(float(durations.mean()), 2) if not durations.empty else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    notes = [
        "Converted one-row-per-thread records into event-study-ready observations.",
        "Event date is anchored on the first SEC staff letter in each thread.",
        "This step does not classify topic or severity; it prepares the clean event base for the next text-feature step.",
    ]
    result = StepResult(
        step_id="04_event_dataset",
        step_name="Thread to Event Dataset",
        status="complete",
        source="data/processed/thread_level and thread_filing_manifest",
        output_files=[
            str(event_raw_csv.relative_to(root)),
            str(event_raw_parquet.relative_to(root)),
            str(event_base_csv.relative_to(root)),
            str(event_base_parquet.relative_to(root)),
            str(preview_path.relative_to(root)),
            str(inclusion_summary_path.relative_to(root)),
            str(summary_path.relative_to(root)),
        ],
        metrics=metrics,
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def build_event_dataset(root: Path | None = None) -> StepResult:
    root = root or project_root()
    thread_df, filing_df = load_inputs(root)
    event_raw_df = build_event_level_raw(thread_df, filing_df)
    return write_outputs(root, event_raw_df)


if __name__ == "__main__":
    result = build_event_dataset()
    print(json.dumps(asdict(result), indent=2))
