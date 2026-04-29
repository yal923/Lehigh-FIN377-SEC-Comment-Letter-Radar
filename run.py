from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"


def load_step_module(filename: str, module_name: str):
    path = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_sp500_reference(refresh: bool = False):
    module = load_step_module("01_sp500_reference.py", "sp500_reference")
    return module.build_sp500_reference(refresh=refresh, root=ROOT)


def run_sec_raw_inventory(copy_from_archive: bool = True):
    module = load_step_module("02_sec_raw_inventory.py", "sec_raw_inventory")
    return module.build_sec_raw_inventory(root=ROOT, copy_from_archive=copy_from_archive)


def run_thread_dataset():
    module = load_step_module("03_thread_dataset.py", "thread_dataset")
    return module.build_thread_dataset(root=ROOT)


def run_event_dataset():
    module = load_step_module("04_event_dataset.py", "event_dataset")
    return module.build_event_dataset(root=ROOT)


def run_text_features():
    module = load_step_module("05_text_features.py", "text_features")
    return module.build_text_features(root=ROOT)


def run_market_data(copy_from_archive: bool = True):
    module = load_step_module("06_market_data.py", "market_data")
    return module.build_market_data(root=ROOT, copy_from_archive=copy_from_archive)


def run_event_study_regression():
    module = load_step_module("07_event_study_regression.py", "event_study_regression")
    return module.build_event_study_regression(root=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEC comment-letter project pipeline steps.")
    parser.add_argument(
        "--step",
        default="sp500",
        choices=["sp500", "sec_raw", "threads", "events", "text_features", "market", "event_study", "all"],
        help="Pipeline step to run. 'all' currently runs implemented official steps only.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh external data when available.")
    parser.add_argument(
        "--no-archive-copy",
        action="store_true",
        help="For cache-backed steps, use existing official data cache without copying from Archive Prototype.",
    )
    args = parser.parse_args()

    results = []
    if args.step in {"sp500", "all"}:
        results.append(run_sp500_reference(refresh=args.refresh))
    if args.step in {"sec_raw", "all"}:
        results.append(run_sec_raw_inventory(copy_from_archive=not args.no_archive_copy))
    if args.step in {"threads", "all"}:
        results.append(run_thread_dataset())
    if args.step in {"events", "all"}:
        results.append(run_event_dataset())
    if args.step in {"text_features", "all"}:
        results.append(run_text_features())
    if args.step in {"market", "all"}:
        results.append(run_market_data(copy_from_archive=not args.no_archive_copy))
    if args.step in {"event_study", "all"}:
        results.append(run_event_study_regression())

    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
