from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
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


def copy_raw_cache_from_archive(root: Path) -> tuple[str, list[str]]:
    source_dir = root / "Archive Prototype" / "data" / "raw" / "sec"
    target_dir = root / "data" / "raw" / "sec"
    notes: list[str] = []

    if not source_dir.exists():
        raise FileNotFoundError(f"Archive SEC raw cache not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_dir,
        target_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"),
    )
    notes.append("Copied available SEC raw cache from Archive Prototype into official data/raw/sec.")
    return "archive_raw_cache", notes


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_files_and_bytes(path: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    for item in path.rglob("*"):
        if item.is_file() and item.name != ".DS_Store":
            file_count += 1
            byte_count += item.stat().st_size
    return file_count, byte_count


def collect_inventory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sec_dir = root / "data" / "raw" / "sec"
    if not sec_dir.exists():
        raise FileNotFoundError(f"Official SEC raw cache not found: {sec_dir}")

    thread_rows: list[dict[str, Any]] = []
    filing_rows: list[dict[str, Any]] = []
    ticker_dirs = sorted(path for path in sec_dir.iterdir() if path.is_dir())

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        company_submissions = ticker_dir / "company_submissions.json"
        submission_archive_count = len(list((ticker_dir / "submission_archives").glob("*.json"))) if (ticker_dir / "submission_archives").exists() else 0
        thread_dirs = sorted(
            path
            for path in ticker_dir.iterdir()
            if path.is_dir() and path.name != "submission_archives"
        )

        if not thread_dirs:
            file_count, byte_count = count_files_and_bytes(ticker_dir)
            thread_rows.append(
                {
                    "ticker": ticker,
                    "thread_id": None,
                    "forms_in_thread": None,
                    "accession_count": 0,
                    "filing_count": 0,
                    "extracted_text_count": 0,
                    "ok_extraction_count": 0,
                    "file_count": file_count,
                    "byte_count": byte_count,
                    "has_company_submissions": company_submissions.exists(),
                    "submission_archive_count": submission_archive_count,
                    "sample_start_date": None,
                    "sample_end_date": None,
                }
            )
            continue

        for thread_dir in thread_dirs:
            metadata_path = thread_dir / "metadata.json"
            manifest_path = thread_dir / "extraction_manifest.json"
            metadata = load_json(metadata_path) if metadata_path.exists() else {}
            manifest = load_json(manifest_path) if manifest_path.exists() else []
            file_count, byte_count = count_files_and_bytes(thread_dir)
            text_count = len(list((thread_dir / "extracted_text").glob("*.txt"))) if (thread_dir / "extracted_text").exists() else 0
            ok_count = sum(1 for row in manifest if row.get("extraction_status") == "ok")
            forms = metadata.get("forms_in_thread") or [row.get("form") for row in manifest if row.get("form")]
            normalized_forms = metadata.get("normalized_forms_in_thread") or [
                row.get("normalized_form") for row in manifest if row.get("normalized_form")
            ]

            thread_rows.append(
                {
                    "ticker": ticker,
                    "thread_id": thread_dir.name,
                    "forms_in_thread": ", ".join(forms) if forms else None,
                    "normalized_forms_in_thread": ", ".join(normalized_forms) if normalized_forms else None,
                    "accession_count": len(metadata.get("accessions", [])) or len(manifest),
                    "filing_count": len(manifest),
                    "extracted_text_count": text_count,
                    "ok_extraction_count": ok_count,
                    "file_count": file_count,
                    "byte_count": byte_count,
                    "has_company_submissions": company_submissions.exists(),
                    "submission_archive_count": submission_archive_count,
                    "sample_start_date": metadata.get("sample_start_date"),
                    "sample_end_date": metadata.get("sample_end_date"),
                }
            )

            for row in manifest:
                filing_rows.append(
                    {
                        "ticker": ticker,
                        "thread_id": thread_dir.name,
                        "accession_number": row.get("accessionNumber"),
                        "filing_date": row.get("filingDate"),
                        "form": row.get("form"),
                        "normalized_form": row.get("normalized_form"),
                        "form_role": row.get("form_role"),
                        "firm_name": row.get("firm_name"),
                        "industry": row.get("industry"),
                        "document_name": row.get("document_name"),
                        "extraction_status": row.get("extraction_status"),
                        "letter_word_count_file": row.get("letter_word_count_file"),
                    }
                )

    thread_df = pd.DataFrame(thread_rows)
    filing_df = pd.DataFrame(filing_rows)
    metrics = {
        "ticker_count": int(thread_df["ticker"].nunique()) if not thread_df.empty else 0,
        "thread_count": int(thread_df["thread_id"].notna().sum()) if "thread_id" in thread_df else 0,
        "filing_count": int(len(filing_df)),
        "company_submission_file_count": int(thread_df.groupby("ticker")["has_company_submissions"].max().sum()) if not thread_df.empty else 0,
        "submission_archive_file_count": int(thread_df.groupby("ticker")["submission_archive_count"].max().sum()) if not thread_df.empty else 0,
        "extracted_text_file_count": int(thread_df["extracted_text_count"].sum()) if not thread_df.empty else 0,
        "ok_extraction_count": int(thread_df["ok_extraction_count"].sum()) if not thread_df.empty else 0,
        "raw_cache_mb": round(sum(path.stat().st_size for path in sec_dir.rglob("*") if path.is_file()) / 1_000_000, 2),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return thread_df, filing_df, metrics


def write_workflow_artifacts(
    root: Path,
    thread_df: pd.DataFrame,
    filing_df: pd.DataFrame,
    metrics: dict[str, Any],
    source: str,
    notes: list[str],
) -> StepResult:
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    thread_inventory_path = workflow_dir / "02_sec_thread_inventory.csv"
    filing_preview_path = workflow_dir / "02_sec_raw_preview.csv"
    form_summary_path = tables_dir / "sec_raw_form_summary.csv"
    summary_path = workflow_dir / "02_sec_raw_summary.json"

    thread_df.to_csv(thread_inventory_path, index=False)
    filing_df.head(100).to_csv(filing_preview_path, index=False)
    if filing_df.empty:
        form_summary = pd.DataFrame(columns=["form", "normalized_form", "form_role", "filing_count"])
    else:
        form_summary = (
            filing_df.groupby(["form", "normalized_form", "form_role"], dropna=False)
            .size()
            .reset_index(name="filing_count")
            .sort_values("filing_count", ascending=False)
        )
    form_summary.to_csv(form_summary_path, index=False)

    result = StepResult(
        step_id="02_sec_raw_inventory",
        step_name="SEC Raw Cache Inventory",
        status="complete",
        source=source,
        output_files=[
            str((root / "data" / "raw" / "sec").relative_to(root)),
            str(thread_inventory_path.relative_to(root)),
            str(filing_preview_path.relative_to(root)),
            str(form_summary_path.relative_to(root)),
            str(summary_path.relative_to(root)),
        ],
        metrics=metrics,
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def build_sec_raw_inventory(root: Path | None = None, copy_from_archive: bool = True) -> StepResult:
    root = root or project_root()
    source = "active_raw_cache"
    notes: list[str] = []
    archive_raw_cache = root / "Archive Prototype" / "data" / "raw" / "sec"
    active_raw_cache = root / "data" / "raw" / "sec"

    if copy_from_archive and archive_raw_cache.exists():
        source, notes = copy_raw_cache_from_archive(root)
    elif active_raw_cache.exists():
        notes.append("Used existing official SEC raw cache in data/raw/sec.")
    elif archive_raw_cache.exists():
        source, notes = copy_raw_cache_from_archive(root)
    else:
        raise FileNotFoundError(
            "SEC raw cache not found. Expected official data/raw/sec or Archive Prototype/data/raw/sec."
        )

    thread_df, filing_df, metrics = collect_inventory(root)
    notes.append("Generated lightweight inventory files for Streamlit display instead of loading full raw cache.")
    return write_workflow_artifacts(root, thread_df, filing_df, metrics, source, notes)


if __name__ == "__main__":
    result = build_sec_raw_inventory()
    print(json.dumps(asdict(result), indent=2))
