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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_file(root: Path, raw_value: Any, fallback_path: Path) -> str:
    candidate: Path | None = None
    if isinstance(raw_value, str) and raw_value.strip():
        value_path = Path(raw_value)
        candidate = value_path if value_path.is_absolute() else root / value_path
    if candidate is None or not candidate.exists():
        candidate = fallback_path
    if not candidate.exists():
        return ""
    return candidate.read_text(encoding="utf-8", errors="replace")


def collect_filing_rows(root: Path) -> pd.DataFrame:
    sec_dir = root / "data" / "raw" / "sec"
    if not sec_dir.exists():
        raise FileNotFoundError(f"Official SEC raw cache not found: {sec_dir}")

    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(sec_dir.glob("*/*/extraction_manifest.json")):
        thread_dir = manifest_path.parent
        manifest = load_json(manifest_path)
        if not isinstance(manifest, list):
            continue

        for row in manifest:
            accession = row.get("accessionNumber")
            fallback_text_path = thread_dir / "extracted_text" / f"{accession}.txt"
            text = read_text_file(root, row.get("extracted_text_path"), fallback_text_path)
            out_row = {
                "ticker": row.get("ticker") or thread_dir.parent.name,
                "thread_id": row.get("thread_id") or thread_dir.name,
                "accession_number": accession,
                "filing_date": row.get("filingDate"),
                "acceptance_datetime": row.get("acceptanceDateTime"),
                "form": row.get("form"),
                "normalized_form": row.get("normalized_form"),
                "form_role": row.get("form_role"),
                "firm_name": row.get("firm_name"),
                "cik": row.get("cik"),
                "industry": row.get("industry"),
                "file_number": row.get("fileNumber"),
                "document_name": row.get("document_name"),
                "document_url": row.get("document_url"),
                "raw_document_path": row.get("raw_document_path"),
                "extracted_text_path": row.get("extracted_text_path"),
                "extraction_status": row.get("extraction_status"),
                "letter_word_count_file": row.get("letter_word_count_file"),
                "letter_text": text,
                "letter_word_count": len(text.split()),
            }
            rows.append(out_row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["acceptance_datetime"] = pd.to_datetime(df["acceptance_datetime"], errors="coerce", utc=True)
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["letter_word_count_file"] = pd.to_numeric(df["letter_word_count_file"], errors="coerce")
    df = df.sort_values(["ticker", "thread_id", "filing_date", "accession_number"]).reset_index(drop=True)
    df["filing_sequence_in_thread"] = df.groupby(["ticker", "thread_id"]).cumcount() + 1
    return df


def join_nonempty(values: pd.Series, separator: str = ", ") -> str:
    return separator.join(str(value) for value in values.dropna().tolist() if str(value).strip())


def first_nonempty(values: pd.Series) -> Any:
    clean = values.dropna()
    if clean.empty:
        return None
    return clean.iloc[0]


def build_thread_level(filing_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if filing_df.empty:
        return pd.DataFrame()

    for (ticker, thread_id), group in filing_df.groupby(["ticker", "thread_id"], sort=True):
        group = group.sort_values(["filing_date", "accession_number"]).copy()
        staff_letters = group[group["normalized_form"] == "STAFF_LETTER"]
        responses = group[group["normalized_form"] == "FILER_RESPONSE"]
        staff_actions = group[group["normalized_form"] == "STAFF_ACTION"]
        main = staff_letters.iloc[0] if not staff_letters.empty else group.iloc[0]
        first_date = group["filing_date"].min()
        last_date = group["filing_date"].max()
        first_staff_date = staff_letters["filing_date"].min() if not staff_letters.empty else pd.NaT
        staff_letter_text = "\n\n".join(staff_letters["letter_text"].fillna("").tolist())
        response_text = "\n\n".join(responses["letter_text"].fillna("").tolist())
        staff_action_text = "\n\n".join(staff_actions["letter_text"].fillna("").tolist())
        all_thread_text = "\n\n".join(group["letter_text"].fillna("").tolist())

        rows.append(
            {
                "ticker": ticker,
                "thread_id": thread_id,
                "firm_name": first_nonempty(group["firm_name"]),
                "cik": first_nonempty(group["cik"]),
                "industry": first_nonempty(group["industry"]),
                "event_anchor_accession": main.get("accession_number"),
                "event_anchor_date": first_staff_date.date().isoformat() if pd.notna(first_staff_date) else None,
                "first_filing_date": first_date.date().isoformat() if pd.notna(first_date) else None,
                "last_filing_date": last_date.date().isoformat() if pd.notna(last_date) else None,
                "thread_duration_days": int((last_date - first_date).days) if pd.notna(first_date) and pd.notna(last_date) else None,
                "n_filings_in_thread": int(len(group)),
                "n_staff_letter": int(len(staff_letters)),
                "n_filer_response": int(len(responses)),
                "n_staff_action": int(len(staff_actions)),
                "forms_in_thread": join_nonempty(group["form"]),
                "normalized_forms_in_thread": join_nonempty(group["normalized_form"]),
                "accessions_json": json.dumps(group["accession_number"].dropna().tolist()),
                "filing_dates_json": json.dumps([date.date().isoformat() for date in group["filing_date"].dropna()]),
                "document_names_json": json.dumps(group["document_name"].dropna().tolist()),
                "document_urls_json": json.dumps(group["document_url"].dropna().tolist()),
                "staff_letter_extraction_statuses": join_nonempty(staff_letters["extraction_status"]),
                "response_extraction_statuses": join_nonempty(responses["extraction_status"]),
                "staff_letter_word_count": int(len(staff_letter_text.split())),
                "response_word_count": int(len(response_text.split())),
                "staff_action_word_count": int(len(staff_action_text.split())),
                "all_thread_word_count": int(len(all_thread_text.split())),
                "staff_letter_text": staff_letter_text,
                "response_text": response_text,
                "staff_action_text": staff_action_text,
                "all_thread_text": all_thread_text,
                "threading_status": "matched" if not staff_letters.empty else "no_staff_letter_anchor",
                "text_status": "ok" if len(staff_letter_text.split()) > 0 else "empty_staff_letter_text",
            }
        )

    return pd.DataFrame(rows).sort_values(["ticker", "event_anchor_date", "thread_id"]).reset_index(drop=True)


def write_outputs(root: Path, filing_df: pd.DataFrame, thread_df: pd.DataFrame) -> StepResult:
    processed_dir = root / "data" / "processed"
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    filing_csv = processed_dir / "thread_filing_manifest.csv"
    filing_parquet = processed_dir / "thread_filing_manifest.parquet"
    thread_csv = processed_dir / "thread_level.csv"
    thread_parquet = processed_dir / "thread_level.parquet"
    filing_preview_path = workflow_dir / "03_thread_filing_preview.csv"
    thread_preview_path = workflow_dir / "03_thread_preview.csv"
    role_summary_path = tables_dir / "thread_role_summary.csv"
    summary_path = workflow_dir / "03_thread_dataset_summary.json"

    filing_df.to_csv(filing_csv, index=False)
    filing_df.to_parquet(filing_parquet, index=False)
    thread_df.to_csv(thread_csv, index=False)
    thread_df.to_parquet(thread_parquet, index=False)

    preview_thread_cols = [
        "ticker",
        "thread_id",
        "firm_name",
        "industry",
        "event_anchor_date",
        "n_filings_in_thread",
        "n_staff_letter",
        "n_filer_response",
        "thread_duration_days",
        "staff_letter_word_count",
        "text_status",
    ]
    preview_filing_cols = [
        "ticker",
        "thread_id",
        "filing_sequence_in_thread",
        "accession_number",
        "filing_date",
        "form",
        "normalized_form",
        "form_role",
        "extraction_status",
        "letter_word_count",
    ]
    thread_df[preview_thread_cols].head(100).to_csv(thread_preview_path, index=False)
    filing_df[preview_filing_cols].head(150).to_csv(filing_preview_path, index=False)

    role_summary = (
        filing_df.groupby(["normalized_form", "form_role"], dropna=False)
        .size()
        .reset_index(name="filing_count")
        .sort_values("filing_count", ascending=False)
    )
    role_summary.to_csv(role_summary_path, index=False)

    durations = pd.to_numeric(thread_df["thread_duration_days"], errors="coerce") if not thread_df.empty else pd.Series(dtype=float)
    metrics = {
        "ticker_count": int(thread_df["ticker"].nunique()) if not thread_df.empty else 0,
        "thread_count": int(len(thread_df)),
        "filing_count": int(len(filing_df)),
        "staff_letter_count": int((filing_df["normalized_form"] == "STAFF_LETTER").sum()) if not filing_df.empty else 0,
        "filer_response_count": int((filing_df["normalized_form"] == "FILER_RESPONSE").sum()) if not filing_df.empty else 0,
        "staff_action_count": int((filing_df["normalized_form"] == "STAFF_ACTION").sum()) if not filing_df.empty else 0,
        "complete_staff_text_thread_count": int((thread_df["text_status"] == "ok").sum()) if not thread_df.empty else 0,
        "mean_thread_duration_days": round(float(durations.mean()), 2) if not durations.empty else None,
        "median_thread_duration_days": round(float(durations.median()), 2) if not durations.empty else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    notes = [
        "Built official filing-level and thread-level datasets from data/raw/sec extraction manifests.",
        "The current baseline uses thread IDs already materialized by the V3 raw cache; this step validates and reshapes them for downstream event construction.",
        "Full processed tables are stored as paired CSV and Parquet files; Streamlit uses lightweight preview tables for display.",
    ]
    result = StepResult(
        step_id="03_thread_dataset",
        step_name="Filing to Thread Dataset",
        status="complete",
        source="data/raw/sec extraction manifests",
        output_files=[
            str(filing_csv.relative_to(root)),
            str(filing_parquet.relative_to(root)),
            str(thread_csv.relative_to(root)),
            str(thread_parquet.relative_to(root)),
            str(filing_preview_path.relative_to(root)),
            str(thread_preview_path.relative_to(root)),
            str(role_summary_path.relative_to(root)),
            str(summary_path.relative_to(root)),
        ],
        metrics=metrics,
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def build_thread_dataset(root: Path | None = None) -> StepResult:
    root = root or project_root()
    filing_df = collect_filing_rows(root)
    thread_df = build_thread_level(filing_df)
    return write_outputs(root, filing_df, thread_df)


if __name__ == "__main__":
    result = build_thread_dataset()
    print(json.dumps(asdict(result), indent=2))
