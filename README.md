# SEC Comment Letter Project

This project turns the validated V3 notebook prototype into a reproducible Streamlit project for SEC comment-letter event-study analysis.

The repository includes the current 100-company, 10-year SEC raw cache and market-data cache, so teammates can run the app and reproduce outputs without redownloading SEC or market data.

## Project Structure

```text
run.py                  # Pipeline entrypoint
app.py                  # Streamlit landing page
code/                   # Numbered pipeline modules
configs/                # Sample and text-feature configs
data/raw/               # Cached source data
data/processed/         # Reusable processed datasets
outputs/                # Tables, figures, and workflow summaries
pages/                  # Streamlit pages
docs/                   # Project notes, decisions, and progress
```

`Archive Prototype/`, `.venv/`, `.envrc`, and `AGENTS.md` are intentionally excluded from the GitHub submission version.

## Setup

Use Python 3.11 if available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Website

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Reproduce the Pipeline

The included data cache lets this run without redownloading the SEC raw cache or market data.

```bash
python run.py --step all
```

Individual steps can also be run:

```bash
python run.py --step sp500
python run.py --step sec_raw
python run.py --step threads
python run.py --step events
python run.py --step text_features
python run.py --step market
python run.py --step event_study
```

## Current Outputs

The final dashboard reads official outputs from:

```text
outputs/tables/
outputs/figures/
data/processed/
```

The main final-result views are:

- Topic CAR explorer
- Regression playground
- Result tables
- Static figures

## Notes

- This is the 100-company, 10-year baseline run.
- A future full S&P 500 run can reuse the same project structure.
- CSV/JSON files are included for human validation.
- Parquet files are included for efficient pandas and Streamlit loading.
