# Project Structure

The official project uses a simple root-level structure:

```text
run.py
app.py
code/
configs/
data/
outputs/
pages/
docs/
Archive Prototype/
```

## Official Pipeline

- `run.py` is the root orchestrator.
- `code/` contains numbered step modules.
- `configs/` stores sample and run-mode configurations.
- `data/raw/` stores source and cached input artifacts.
- `data/processed/` stores reusable analytical datasets.
- `outputs/` stores presentation-ready tables, figures, and workflow metadata.
- CSV and JSON are the human-validation formats. Parquet may be kept as an efficient pandas/Streamlit cache when paired with a readable CSV output.
- SEC raw data is shared across samples in `data/raw/sec/`. Sample-specific analyses should be controlled by config files rather than duplicated raw SEC caches.
- Thread-level processed data is materialized in paired files such as `data/processed/thread_filing_manifest.csv` / `.parquet` and `data/processed/thread_level.csv` / `.parquet`.
- Event-level processed data is materialized in paired files such as `data/processed/event_level_raw.csv` / `.parquet` and `data/processed/event_level_base.csv` / `.parquet`.
- Text-feature configuration is stored in `configs/text_feature_config.json`; the featured event dataset is materialized as `data/processed/event_level.csv` / `.parquet`.
- Market raw cache is stored in `data/raw/market/`; normalized daily market rows are materialized as `data/processed/market_data.csv` / `.parquet`.
- Event-study and regression outputs are materialized as `data/processed/event_time.csv` / `.parquet`, `data/processed/regression_dataset.csv` / `.parquet`, `outputs/tables/`, and `outputs/figures/`.

## Streamlit Presentation

- `app.py` is the landing page.
- `pages/01_Background_and_Methods.py` presents the ordered data process flow as horizontal tabs.
- `pages/02_Final_Results.py` presents official current result artifacts from `outputs/` and `data/processed/`.
- `pages/03_Research_Design.py` presents the research question and empirical design.

Each workflow step should expose:

- input files;
- processing logic;
- output files;
- preview tables, usually CSV-backed;
- QA notes and limitations.

## Prototype Archive

`Archive Prototype/` is retained for reference. Official code should be extracted into `code/` and run through `run.py` rather than importing notebook logic directly from the archive.
