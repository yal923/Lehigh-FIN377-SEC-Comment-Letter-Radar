from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import yfinance as yf


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
        df = pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)
    elif csv_files:
        df = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)
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
    return (
        df.dropna(subset=["date", "ticker", "return"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def required_market_bounds(events: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    event_start = int(config.get("event_window_start", -150))
    event_end = int(config.get("event_window_end", 60))
    start_buffer_days = abs(event_start) * 2 + 30
    end_buffer_days = max(event_end, 0) * 2 + 30
    event_dates = pd.to_datetime(events["event_date"], errors="coerce").dropna()
    return (
        event_dates.min() - pd.Timedelta(days=start_buffer_days),
        event_dates.max() + pd.Timedelta(days=end_buffer_days),
    )


def required_market_bounds_by_ticker(
    events: pd.DataFrame,
    config: dict[str, Any],
    benchmark_ticker: str,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    global_start, global_end = required_market_bounds(events, config)
    event_start = int(config.get("event_window_start", -150))
    event_end = int(config.get("event_window_end", 60))
    start_buffer_days = abs(event_start) * 2 + 30
    end_buffer_days = max(event_end, 0) * 2 + 30
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {benchmark_ticker: (global_start, global_end)}
    temp = events[["ticker", "event_date"]].copy()
    temp["event_date"] = pd.to_datetime(temp["event_date"], errors="coerce")
    temp = temp.dropna(subset=["ticker", "event_date"])
    for ticker, group in temp.groupby("ticker"):
        bounds[str(ticker)] = (
            group["event_date"].min() - pd.Timedelta(days=start_buffer_days),
            group["event_date"].max() + pd.Timedelta(days=end_buffer_days),
        )
    return bounds


def cache_covers_required_window(
    market_data: pd.DataFrame,
    required_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    coverage = market_data[market_data["ticker"].isin(required_bounds)].groupby("ticker")["date"].agg(["min", "max"])
    if set(required_bounds) - set(coverage.index):
        return False
    return all(
        coverage.loc[ticker, "min"] <= start and coverage.loc[ticker, "max"] >= end
        for ticker, (start, end) in required_bounds.items()
    )


def download_market_prices(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    download_end = end_date + pd.Timedelta(days=1)
    raw = yf.download(
        tickers=tickers,
        start=start_date.date().isoformat(),
        end=download_end.date().isoformat(),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned an empty market dataset.")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            ticker_raw = raw[ticker].copy()
        else:
            ticker_raw = raw.copy()
        close_col = "Close" if "Close" in ticker_raw.columns else "Adj Close"
        if close_col not in ticker_raw.columns:
            continue
        frame = ticker_raw[[close_col, "Volume"]].reset_index()
        frame.columns = ["date", "close", "volume"]
        frame["ticker"] = ticker
        frame["return"] = frame["close"].pct_change()
        frames.append(frame[["date", "ticker", "close", "volume", "return"]])

    if not frames:
        raise RuntimeError("No usable yfinance price columns were returned.")
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    return prices.dropna(subset=["date", "ticker", "return"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def write_market_cache(root: Path, prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[Path, Path]:
    market_dir = root / "data" / "raw" / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    tickers_hash = hashlib.sha1("|".join(sorted(prices["ticker"].unique())).encode("utf-8")).hexdigest()[:12]
    stem = f"prices_{start_date.date().isoformat()}_{end_date.date().isoformat()}_{tickers_hash}"
    csv_path = market_dir / f"{stem}.csv"
    parquet_path = market_dir / f"{stem}.parquet"
    prices.to_csv(csv_path, index=False)
    prices.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


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
    required_start, required_end = required_market_bounds(events, config)
    required_bounds = required_market_bounds_by_ticker(events, config, benchmark_ticker)
    if not cache_covers_required_window(raw_market, required_bounds):
        refreshed_market = download_market_prices(required_tickers, required_start, required_end)
        cache_csv, cache_parquet = write_market_cache(root, refreshed_market, required_start, required_end)
        raw_market = read_cached_market_prices(root)
        source = "refreshed_yfinance_cache"
        notes.append(
            "Extended market cache with yfinance because the official cache did not cover the configured pre-event volatility window."
        )
        notes.append(f"Added cache files: {cache_csv.relative_to(root)}, {cache_parquet.relative_to(root)}")

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
            "required_start_date": required_start.date().isoformat(),
            "required_end_date": required_end.date().isoformat(),
            "start_date": market_data["date"].min().date().isoformat() if not market_data.empty else None,
            "end_date": market_data["date"].max().date().isoformat() if not market_data.empty else None,
            "raw_market_cache_mb": raw_market_mb,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    notes.extend(
        [
            "Market prices are loaded from official raw cache and refreshed with yfinance only if the configured event/pre-event window requires additional coverage.",
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
