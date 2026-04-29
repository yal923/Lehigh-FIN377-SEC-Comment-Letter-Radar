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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_market_cache_from_archive(root: Path) -> tuple[str, list[str]]:
    source_dir = root / "Archive Prototype" / "data" / "raw" / "market"
    target_dir = root / "data" / "raw" / "market"
    if not source_dir.exists():
        raise FileNotFoundError(f"Archive market cache not found: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_dir,
        target_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"),
    )
    return "archive_market_cache", ["Copied cached market price files from Archive Prototype into official data/raw/market."]


def read_event_level(root: Path) -> pd.DataFrame:
    parquet_path = root / "data" / "processed" / "event_level.parquet"
    csv_path = root / "data" / "processed" / "event_level.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing event-level dataset with text features: {parquet_path}")


def read_cached_market_prices(root: Path) -> pd.DataFrame:
    market_dir = root / "data" / "raw" / "market"
    parquet_files = sorted(market_dir.glob("prices_*.parquet"))
    csv_files = sorted(market_dir.glob("prices_*.csv"))
    if parquet_files:
        df = pd.read_parquet(parquet_files[-1])
    elif csv_files:
        df = pd.read_csv(csv_files[-1])
    else:
        raise FileNotFoundError(f"No cached market price files found in {market_dir}")

    required = ["date", "ticker", "close", "volume", "return"]
    missing = set(required) - set(df.columns)
    if missing:
        raise RuntimeError(f"Unexpected market data schema. Missing columns: {sorted(missing)}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    return df.dropna(subset=["date", "ticker", "return"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_market_data(root: Path, copy_from_archive: bool = True) -> StepResult:
    source = "active_market_cache"
    notes: list[str] = []
    archive_market_cache = root / "Archive Prototype" / "data" / "raw" / "market"
    active_market_cache = root / "data" / "raw" / "market"

    if copy_from_archive and archive_market_cache.exists():
        source, notes = copy_market_cache_from_archive(root)
    elif active_market_cache.exists():
        notes.append("Used existing official market cache in data/raw/market.")
    elif archive_market_cache.exists():
        source, notes = copy_market_cache_from_archive(root)
    else:
        raise FileNotFoundError(
            "Market cache not found. Expected official data/raw/market or Archive Prototype/data/raw/market."
        )

    config = load_json(root / "configs" / "sample_100_10y.json")
    benchmark_ticker = config.get("benchmark_ticker", "SPY")
    events = read_event_level(root)
    event_tickers = sorted(events["ticker"].dropna().astype(str).unique().tolist())
    required_tickers = sorted(set(event_tickers) | {benchmark_ticker})

    raw_market = read_cached_market_prices(root)
    market_data = raw_market[raw_market["ticker"].isin(required_tickers)].copy()
    market_data["is_benchmark"] = market_data["ticker"].eq(benchmark_ticker)
    market_data["is_event_ticker"] = market_data["ticker"].isin(event_tickers)

    processed_dir = root / "data" / "processed"
    workflow_dir = root / "outputs" / "workflow"
    tables_dir = root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    market_csv = processed_dir / "market_data.csv"
    market_parquet = processed_dir / "market_data.parquet"
    preview_path = workflow_dir / "06_market_data_preview.csv"
    coverage_path = tables_dir / "market_coverage_summary.csv"
    summary_path = workflow_dir / "06_market_data_summary.json"

    market_data.to_csv(market_csv, index=False)
    market_data.to_parquet(market_parquet, index=False)
    market_data.head(150).to_csv(preview_path, index=False)

    coverage = (
        market_data.groupby("ticker")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            trading_days=("date", "count"),
            mean_return=("return", "mean"),
            return_volatility=("return", "std"),
            is_benchmark=("is_benchmark", "max"),
            is_event_ticker=("is_event_ticker", "max"),
        )
        .reset_index()
        .sort_values(["is_benchmark", "ticker"], ascending=[False, True])
    )
    coverage.to_csv(coverage_path, index=False)

    covered_event_tickers = set(coverage.loc[coverage["is_event_ticker"], "ticker"])
    missing_event_tickers = sorted(set(event_tickers) - covered_event_tickers)
    raw_market_dir = root / "data" / "raw" / "market"
    raw_market_mb = round(
        sum(path.stat().st_size for path in raw_market_dir.rglob("*") if path.is_file()) / 1_000_000,
        2,
    )
    metrics = {
        "row_count": int(len(market_data)),
        "ticker_count": int(market_data["ticker"].nunique()),
        "event_ticker_count": int(len(event_tickers)),
        "event_ticker_covered_count": int(len(covered_event_tickers)),
        "event_ticker_missing_count": int(len(missing_event_tickers)),
        "benchmark_ticker": benchmark_ticker,
        "start_date": market_data["date"].min().date().isoformat() if not market_data.empty else None,
        "end_date": market_data["date"].max().date().isoformat() if not market_data.empty else None,
        "raw_market_cache_mb": raw_market_mb,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    notes.extend(
        [
            "Market prices are imported from the archived yfinance cache for reproducibility; no live market download is performed in this baseline step.",
            "The dataset includes event tickers plus the configured benchmark ticker for later abnormal-return construction.",
            f"Missing event tickers: {missing_event_tickers}" if missing_event_tickers else "All event tickers are covered by the market cache.",
        ]
    )

    result = StepResult(
        step_id="06_market_data",
        step_name="Market Data",
        status="complete",
        source=source,
        output_files=[
            str((root / "data" / "raw" / "market").relative_to(root)),
            str(market_csv.relative_to(root)),
            str(market_parquet.relative_to(root)),
            str(preview_path.relative_to(root)),
            str(coverage_path.relative_to(root)),
            str(summary_path.relative_to(root)),
        ],
        metrics=metrics,
        notes=notes,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = build_market_data()
    print(json.dumps(asdict(result), indent=2))
