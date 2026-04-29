# Decisions

## Active Decisions

### 2026-04-29: Use Shared SEC Raw Cache With Sample-Level Run Modes

Decision: The official project stores SEC raw files in one shared `data/raw/sec/` cache. The current baseline is the 100-company 10-year sample, configured in `configs/sample_100_10y.json`. A future full S&P 500 10-year run is an optional extension and should coexist with, not replace, the 100-company baseline.

Rationale: SEC comment-letter raw data is comparatively stable and does not need to be redownloaded for the current baseline. A shared raw cache avoids duplicating raw SEC files across sample versions. If a full S&P 500 cache is later built, the 100-company analysis can be sampled from that same raw cache using configuration rather than redownloading or maintaining separate raw-data trees.

Implication: Frontend result pages may later expose sample/run selectors, but raw SEC files should remain shared. Pipeline stages should use sample configs to choose which tickers/events enter a particular analysis.

### 2026-04-29: Limit Regression Playground Controls

Decision: The interactive regression view should focus on `topic` and `severity_score` as main variables, with only `industry` and `pre_volatility` as playable controls for the current design.

Rationale: The current sample size is not large enough to support an unconstrained control-variable playground. The selected controls have clear financial interpretation and are already part of the project concept.

Implication: Planned regression specifications should include combinations such as topic/severity only, plus industry, plus pre-volatility, and plus both controls. Do not introduce new control datasets before a separate decision.

### 2026-04-29: Use Numbered Official Pipeline and Workflow Presentation App

Decision: The official project will use root `run.py`, numbered modules in `code/`, runtime datasets in `data/`, final and workflow presentation artifacts in `outputs/`, and Streamlit pages in `app.py` plus `pages/`.

Rationale: The final presentation needs to show the full data process flow step by step, with each step exposing input data, processing logic, output artifacts, pandas/table previews, and QA notes. This keeps the official project separate from `Archive Prototype/` while preserving the prototype as reference material.

Implication: Future modules should follow the numbered step pattern and write both reusable data artifacts and presentation-ready workflow artifacts.

### 2026-04-29: Keep CSV/JSON for Human Validation and Parquet for Efficient App Loading

Decision: Official data outputs may be written in paired formats when useful: CSV for human-readable table validation, JSON for workflow metadata and summaries, and Parquet for efficient pandas/Streamlit loading.

Rationale: CSV and JSON make outputs easy to inspect and verify manually. Parquet is less familiar to read directly, but it is efficient and reliable for Python data loading, especially as datasets grow.

Implication: Streamlit pages should prefer Parquet for full-table loading when a paired Parquet file exists, while also exposing CSV previews and JSON summaries for human validation.
