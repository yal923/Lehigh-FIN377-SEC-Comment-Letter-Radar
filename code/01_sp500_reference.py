from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Lehigh University yal923@lehigh.edu"}


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


def clean_cell_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_sp500_table(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(
            "Unexpected S&P 500 table schema. "
            f"Missing columns: {sorted(missing)}. Observed columns: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "firm_name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Headquarters Location": "headquarters_location",
            "Date added": "date_added",
            "CIK": "cik",
            "Founded": "founded",
        }
    )
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["source_url"] = WIKI_URL
    df["source_name"] = "Wikipedia List of S&P 500 companies"

    columns = [
        "ticker",
        "firm_name",
        "gics_sector",
        "gics_sub_industry",
        "headquarters_location",
        "date_added",
        "cik",
        "founded",
        "source_name",
        "source_url",
    ]
    return df[columns].sort_values("ticker").reset_index(drop=True)


def fetch_sp500_from_wikipedia(timeout: int = 30) -> pd.DataFrame:
    response = requests.get(WIKI_URL, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError(f"Could not find constituents table at {WIKI_URL}")

    header_row = table.find("tr")
    if header_row is None:
        raise RuntimeError(f"Could not find header row in constituents table at {WIKI_URL}")

    headers = [clean_cell_text(th.get_text(" ", strip=True)) for th in header_row.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [clean_cell_text(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if cells:
            if len(cells) != len(headers):
                raise RuntimeError(
                    "Unexpected S&P 500 table row width. "
                    f"Expected {len(headers)} cells, got {len(cells)} for row: {cells}"
                )
            rows.append(cells)
    return normalize_sp500_table(pd.DataFrame(rows, columns=headers))


def load_archive_reference(root: Path) -> pd.DataFrame:
    archive_path = root / "Archive Prototype" / "data" / "raw" / "reference" / "sp500_constituents.csv"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive reference file not found: {archive_path}")
    return pd.read_csv(archive_path, dtype={"cik": str})


def write_workflow_artifacts(root: Path, df: pd.DataFrame, source: str, notes: list[str]) -> StepResult:
    data_dir = root / "data" / "raw" / "reference"
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "sp500_constituents.csv"
    parquet_path = data_dir / "sp500_constituents.parquet"
    preview_path = workflow_dir / "01_sp500_reference_preview.csv"
    summary_path = workflow_dir / "01_sp500_reference_summary.json"
    sector_path = tables_dir / "sp500_sector_summary.csv"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    df.head(50).to_csv(preview_path, index=False)

    sector_summary = (
        df.groupby("gics_sector", dropna=False)
        .size()
        .reset_index(name="company_count")
        .sort_values(["company_count", "gics_sector"], ascending=[False, True])
    )
    sector_summary.to_csv(sector_path, index=False)

    result = StepResult(
        step_id="01_sp500_reference",
        step_name="S&P 500 Reference Data",
        status="complete",
        source=source,
        output_files=[
            str(csv_path.relative_to(root)),
            str(parquet_path.relative_to(root)),
            str(preview_path.relative_to(root)),
            str(summary_path.relative_to(root)),
            str(sector_path.relative_to(root)),
        ],
        metrics={
            "company_count": int(len(df)),
            "sector_count": int(df["gics_sector"].nunique(dropna=True)),
            "unique_ticker_count": int(df["ticker"].nunique(dropna=True)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def build_sp500_reference(refresh: bool = False, root: Path | None = None) -> StepResult:
    root = root or project_root()
    active_csv = root / "data" / "raw" / "reference" / "sp500_constituents.csv"
    notes: list[str] = []

    if active_csv.exists() and not refresh:
        df = pd.read_csv(active_csv, dtype={"cik": str})
        source = "active_cache"
        notes.append("Loaded existing active S&P 500 reference file.")
    else:
        try:
            df = fetch_sp500_from_wikipedia()
            source = "wikipedia_live"
            notes.append("Fetched current S&P 500 constituents from Wikipedia.")
        except Exception as exc:
            df = load_archive_reference(root)
            source = "archive_cache"
            notes.append(f"Live Wikipedia fetch failed; used archived reference snapshot. Error: {exc}")

    return write_workflow_artifacts(root=root, df=df, source=source, notes=notes)


if __name__ == "__main__":
    result = build_sp500_reference(refresh=True)
    print(json.dumps(asdict(result), indent=2))
