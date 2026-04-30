from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "outputs" / "workflow"
TABLES_DIR = ROOT / "outputs" / "tables"
DATA_DIR = ROOT / "data"


st.set_page_config(page_title="Background & Methods", layout="wide")
st.title("SEC Comment Letter Evaluation: A Step-by-Step Guide")
st.caption("A transparent walkthrough from source data to event-study and regression outputs.")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(csv_path: Path, parquet_path: Path | None = None, **kwargs) -> pd.DataFrame:
    if parquet_path is not None and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path, **kwargs)


tabs = st.tabs(
    [
        "(1) S&P 500 Reference",
        "(2) SEC Raw Download",
        "(3) Filing to Thread",
        "(4) Event Dataset",
        "(5) Text Features",
        "(6) Market Data",
        "(7) Event Study & Regression",
    ]
)


with tabs[0]:
    summary_path = WORKFLOW_DIR / "01_sp500_reference_summary.json"
    full_csv_path = DATA_DIR / "raw" / "reference" / "sp500_constituents.csv"
    full_parquet_path = DATA_DIR / "raw" / "reference" / "sp500_constituents.parquet"

    st.header("S&P 500 Reference Data")
    st.write("This table is the company universe used by later sampling and SEC matching steps.")

    if not summary_path.exists() or not full_csv_path.exists():
        st.warning("Step 1 artifacts are not available yet. Run `./.venv/bin/python run.py --step sp500`.")
    else:
        reference_df = read_table(full_csv_path, full_parquet_path, dtype={"cik": str})
        sectors = sorted(reference_df["gics_sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("GICS sector", ["All sectors"] + sectors)

        filtered_reference = reference_df.copy()
        if selected_sector != "All sectors":
            filtered_reference = filtered_reference[filtered_reference["gics_sector"] == selected_sector]

        sub_industries = sorted(filtered_reference["gics_sub_industry"].dropna().unique().tolist())
        selected_sub_industry = st.selectbox(
            "GICS sub-industry",
            ["All sub-industries"] + sub_industries,
        )
        if selected_sub_industry != "All sub-industries":
            filtered_reference = filtered_reference[
                filtered_reference["gics_sub_industry"] == selected_sub_industry
            ]

        st.subheader("Full Reference Table")
        st.caption(f"Showing {len(filtered_reference):,} of {len(reference_df):,} S&P 500 companies.")
        st.dataframe(filtered_reference, use_container_width=True, hide_index=True)
        st.caption(
            "Streamlit loads the Parquet cache when available, with CSV as the human-readable companion. "
            "To inspect this table in the project folder, use `data/raw/reference/sp500_constituents.parquet` "
            "or `data/raw/reference/sp500_constituents.csv`."
        )


with tabs[1]:
    summary_path = WORKFLOW_DIR / "02_sec_raw_summary.json"
    thread_inventory_path = WORKFLOW_DIR / "02_sec_thread_inventory.csv"
    form_summary_path = TABLES_DIR / "sec_raw_form_summary.csv"

    st.header("SEC Raw Cache Inventory")
    st.write(
        "This step registers the official SEC raw cache and turns it into a lightweight thread-level "
        "inventory for downstream processing."
    )

    if not summary_path.exists() or not thread_inventory_path.exists():
        st.warning("Step 2 artifacts are not available yet. Run `./.venv/bin/python run.py --step sec_raw`.")
    else:
        summary = load_json(summary_path)
        metrics = summary["metrics"]
        extraction_success_rate = (
            metrics["ok_extraction_count"] / metrics["extracted_text_file_count"]
            if metrics["extracted_text_file_count"]
            else 0
        )
        summary_table = pd.DataFrame(
            [
                {
                    "tickers_covered": metrics["ticker_count"],
                    "raw_filings": metrics["filing_count"],
                    "raw_cache_size_mb": metrics["raw_cache_mb"],
                    "extracted_text_files": metrics["extracted_text_file_count"],
                    "extraction_success_rate": f"{extraction_success_rate:.1%}",
                }
            ]
        )

        st.subheader("SEC Raw Cache Summary")
        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True,
        )

        inventory_df = pd.read_csv(thread_inventory_path)
        filtered_inventory = inventory_df.copy()

        tickers = sorted(inventory_df["ticker"].dropna().unique().tolist())
        selected_ticker = st.selectbox("Ticker", ["All tickers"] + tickers)
        if selected_ticker != "All tickers":
            filtered_inventory = filtered_inventory[filtered_inventory["ticker"] == selected_ticker]

        normalized_forms = sorted(
            {
                form.strip()
                for forms in inventory_df["normalized_forms_in_thread"].dropna()
                for form in str(forms).split(",")
                if form.strip()
            }
        )
        selected_form = st.selectbox("Normalized form in thread", ["All forms"] + normalized_forms)
        if selected_form != "All forms":
            filtered_inventory = filtered_inventory[
                filtered_inventory["normalized_forms_in_thread"].str.contains(selected_form, na=False)
            ]

        submission_filter = st.selectbox("Company submissions", ["All", "Yes", "No"])
        if submission_filter != "All":
            has_submission = submission_filter == "Yes"
            filtered_inventory = filtered_inventory[
                filtered_inventory["has_company_submissions"] == has_submission
            ]

        st.subheader("SEC Raw Inventory Table")
        st.caption(f"Showing {len(filtered_inventory):,} of {len(inventory_df):,} SEC raw-cache threads.")
        st.dataframe(filtered_inventory, use_container_width=True, hide_index=True)
        st.caption(
            "Raw SEC correspondence files are stored under `data/raw/sec/`. "
            "The workflow-level inventory table is stored at `outputs/workflow/02_sec_thread_inventory.csv`."
        )
        st.caption(
            "`thread_id` is materialized during raw-cache construction from SEC correspondence metadata, "
            "company identifiers, chronology, and the anchor staff-letter accession. The next step uses this "
            "identifier to reshape filing rows into one-row-per-thread observations."
        )


with tabs[2]:
    summary_path = WORKFLOW_DIR / "03_thread_dataset_summary.json"
    role_summary_path = TABLES_DIR / "thread_role_summary.csv"
    raw_inventory_path = WORKFLOW_DIR / "02_sec_thread_inventory.csv"
    full_thread_csv_path = DATA_DIR / "processed" / "thread_level.csv"
    full_thread_parquet_path = DATA_DIR / "processed" / "thread_level.parquet"

    st.header("Filing to Thread")
    st.write(
        "This step converts single SEC correspondence filings into one-row-per-thread records. "
        "A thread starts with a staff letter and keeps company responses and follow-up staff letters "
        "as context for the same regulatory review."
    )

    if not summary_path.exists():
        st.warning("Step 3 artifacts are not available yet. Run `./.venv/bin/python run.py --step threads`.")
    else:
        summary = load_json(summary_path)
        metrics = summary["metrics"]
        summary_table = pd.DataFrame(
            [
                {
                    "thread_tickers": metrics["ticker_count"],
                    "filing_rows": metrics["filing_count"],
                    "review_threads": metrics["thread_count"],
                    "mean_duration_days": metrics["mean_thread_duration_days"],
                    "median_duration_days": metrics["median_thread_duration_days"],
                }
            ]
        )

        st.subheader("Filing to Thread Summary")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        full_thread_df = read_table(full_thread_csv_path, full_thread_parquet_path)

        display_cols = [
            "ticker",
            "thread_id",
            "firm_name",
            "industry",
            "event_anchor_date",
            "first_filing_date",
            "last_filing_date",
            "n_filings_in_thread",
            "n_staff_letter",
            "n_filer_response",
            "thread_duration_days",
            "staff_letter_word_count",
            "response_word_count",
            "text_status",
        ]
        thread_display = full_thread_df[display_cols].copy()

        tickers = sorted(thread_display["ticker"].dropna().unique().tolist())
        selected_thread_ticker = st.selectbox("Thread ticker", ["All tickers"] + tickers)
        if selected_thread_ticker != "All tickers":
            thread_display = thread_display[thread_display["ticker"] == selected_thread_ticker]

        industries = sorted(thread_display["industry"].dropna().unique().tolist())
        selected_thread_industry = st.selectbox("Thread industry", ["All industries"] + industries)
        if selected_thread_industry != "All industries":
            thread_display = thread_display[thread_display["industry"] == selected_thread_industry]

        text_statuses = sorted(thread_display["text_status"].dropna().unique().tolist())
        selected_text_status = st.selectbox("Text status", ["All statuses"] + text_statuses)
        if selected_text_status != "All statuses":
            thread_display = thread_display[thread_display["text_status"] == selected_text_status]

        st.subheader("Thread-Level Dataset")
        st.caption(f"Showing {len(thread_display):,} of {len(full_thread_df):,} SEC review threads.")
        st.dataframe(thread_display, use_container_width=True, hide_index=True)
        st.caption(
            "The app loads the Parquet cache when available; the paired CSV remains available for manual inspection. "
            "To inspect this table in the project folder, use `data/processed/thread_level.parquet` "
            "or `data/processed/thread_level.csv`."
        )

        st.subheader("Filing Role Summary")
        st.write("This table explains how filing rows are classified before they are grouped into threads.")
        st.dataframe(pd.read_csv(role_summary_path), use_container_width=True, hide_index=True)

        coverage_note = (
            "The raw cache starts from 100 tickers, but the official thread-level dataset contains 93 tickers. "
            "The excluded tickers have raw cache folders but no usable extracted filing rows that can be grouped into review threads."
        )
        if raw_inventory_path.exists():
            raw_inventory_df = pd.read_csv(raw_inventory_path)
            missing_tickers = sorted(
                set(raw_inventory_df["ticker"].dropna()) - set(full_thread_df["ticker"].dropna())
            )
            if missing_tickers:
                coverage_note += f" In the current baseline, those tickers are: {', '.join(missing_tickers)}."
        with st.expander("Notes"):
            st.markdown(
                f"""
**Ticker coverage.** {coverage_note}

**Thread ID and membership.** `thread_id` comes from the V3 raw cache directory and extraction manifest.
It uses the ticker plus the anchor staff-letter accession pattern, such as `RCL_0000000000-19-007772`.
Step 3 groups filing rows by `ticker + thread_id` to create one row per SEC review thread.

Thread membership is usually identifiable from SEC correspondence signals: the same company, CIK, or ticker;
the same SEC file number; a continuous filing sequence where a staff letter is followed by a company response;
the typical document pattern of `UPLOAD` staff letter, `CORRESP` company response, and possible `UPLOAD` follow-up;
accession or folder records that are returned together in SEC correspondence search; and letter text that may
reference the same filing, review, prior comment, or response date. The official Step 3 does not re-infer that
membership from scratch; it validates and reshapes the already-materialized thread grouping.

**Thread duration.** `thread_duration_days` measures the days from the first filing in a thread to the last
related filing in that same review conversation.
"""
            )


with tabs[3]:
    summary_path = WORKFLOW_DIR / "04_event_dataset_summary.json"
    full_event_csv_path = DATA_DIR / "processed" / "event_level_base.csv"
    full_event_parquet_path = DATA_DIR / "processed" / "event_level_base.parquet"

    st.header("Event Dataset")
    st.write(
        "This step turns one-row-per-thread records into event-study-ready observations. "
        "The event date is anchored on the first SEC staff letter in the thread."
    )

    if not summary_path.exists():
        st.warning("Step 4 artifacts are not available yet. Run `./.venv/bin/python run.py --step events`.")
    else:
        summary = load_json(summary_path)
        metrics = summary["metrics"]
        summary_table = pd.DataFrame(
            [
                {
                    "raw_events": metrics["raw_event_count"],
                    "included_events": metrics["included_event_count"],
                    "excluded_events": metrics["excluded_event_count"],
                    "tickers": metrics["ticker_count"],
                    "industries": metrics["industry_count"],
                    "duplicate_event_ids": metrics["duplicate_event_id_count"],
                }
            ]
        )

        st.subheader("Event Dataset Summary")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        full_event_df = read_table(full_event_csv_path, full_event_parquet_path)
        display_cols = [
            "event_id",
            "event_date",
            "ticker",
            "firm_name",
            "industry",
            "thread_id",
            "event_anchor_accession",
            "event_construction_method",
            "n_filings_in_thread",
            "thread_round_count",
            "thread_duration_days",
            "staff_letter_word_count",
            "event_inclusion_status",
        ]
        event_display = full_event_df[display_cols].copy()

        event_tickers = sorted(event_display["ticker"].dropna().unique().tolist())
        selected_event_ticker = st.selectbox("Event ticker", ["All tickers"] + event_tickers)
        if selected_event_ticker != "All tickers":
            event_display = event_display[event_display["ticker"] == selected_event_ticker]

        event_industries = sorted(event_display["industry"].dropna().unique().tolist())
        selected_event_industry = st.selectbox("Event industry", ["All industries"] + event_industries)
        if selected_event_industry != "All industries":
            event_display = event_display[event_display["industry"] == selected_event_industry]

        st.subheader("Event-Level Dataset")
        st.caption(f"Showing {len(event_display):,} of {len(full_event_df):,} included SEC events.")
        st.dataframe(event_display, use_container_width=True, hide_index=True)
        st.caption(
            "This is the analysis base before adding topic, severity, market returns, CAR windows, and regression fields. "
            "To inspect this table in the project folder, use `data/processed/event_level_base.parquet` "
            "or `data/processed/event_level_base.csv`."
        )

        with st.expander("Notes"):
            st.markdown(
                """
**What changes from thread to event.** Step 3 creates one row per SEC review thread. Step 4 keeps that
one-row structure, but reframes each included thread as an event-study observation with a stable `event_id`,
an `event_date`, and an inclusion status.

**Event date.** `event_date` is anchored on the first SEC staff letter in the thread. This is the date used
by later market-data, CAR, and regression steps.

**Event ID.** `event_id` is built from the ticker plus the anchor staff-letter accession, so downstream
tables can join on one stable analysis key.

**Scope.** This step does not classify topic or severity and does not add market returns. It prepares the
clean event base for the next text-feature and event-study steps.
"""
            )


with tabs[4]:
    summary_path = WORKFLOW_DIR / "05_text_features_summary.json"
    topic_summary_path = TABLES_DIR / "topic_summary.csv"
    severity_summary_path = TABLES_DIR / "severity_bucket_summary.csv"
    full_event_csv_path = DATA_DIR / "processed" / "event_level.csv"
    full_event_parquet_path = DATA_DIR / "processed" / "event_level.parquet"

    st.header("Text Features")
    st.write(
        "This step applies the V3 grouped rule-based topic classifier and computes severity features "
        "from staff-letter text and thread structure."
    )

    if not summary_path.exists():
        st.warning("Step 5 artifacts are not available yet. Run `./.venv/bin/python run.py --step text_features`.")
    else:
        summary = load_json(summary_path)
        metrics = summary["metrics"]
        summary_table = pd.DataFrame(
            [
                {
                    "events": metrics["event_count"],
                    "topic_groups": metrics["topic_count"],
                    "detail_labels": metrics["topic_detail_count"],
                    "severity_buckets": metrics["severity_bucket_count"],
                    "mean_severity_score": metrics["mean_severity_score"],
                    "classifier_version": metrics["classifier_version"],
                }
            ]
        )

        st.subheader("Text Feature Summary")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        full_event_df = read_table(full_event_csv_path, full_event_parquet_path)
        display_cols = [
            "event_id",
            "event_date",
            "ticker",
            "industry",
            "topic",
            "topic_detail",
            "topic_classifier_version",
            "severity_score",
            "severity_bucket",
            "letter_word_count",
            "intensity_term_count",
            "thread_round_count",
            "amendment_indicator",
        ]
        text_feature_display = full_event_df[display_cols].copy()

        text_topics = sorted(text_feature_display["topic"].dropna().unique().tolist())
        selected_text_topic = st.selectbox("Topic", ["All topics"] + text_topics)
        if selected_text_topic != "All topics":
            text_feature_display = text_feature_display[text_feature_display["topic"] == selected_text_topic]

        severity_buckets = sorted(text_feature_display["severity_bucket"].dropna().unique().tolist())
        selected_severity_bucket = st.selectbox("Severity bucket", ["All buckets"] + severity_buckets)
        if selected_severity_bucket != "All buckets":
            text_feature_display = text_feature_display[
                text_feature_display["severity_bucket"] == selected_severity_bucket
            ]

        text_industries = sorted(text_feature_display["industry"].dropna().unique().tolist())
        selected_text_industry = st.selectbox("Text feature industry", ["All industries"] + text_industries)
        if selected_text_industry != "All industries":
            text_feature_display = text_feature_display[text_feature_display["industry"] == selected_text_industry]

        text_ticker_query = st.text_input("Ticker search", "")
        if text_ticker_query.strip():
            text_feature_display = text_feature_display[
                text_feature_display["ticker"].astype(str).str.contains(text_ticker_query.strip(), case=False, na=False)
            ]

        st.subheader("Event-Level Dataset with Text Features")
        st.caption(f"Showing {len(text_feature_display):,} of {len(full_event_df):,} events with text features.")
        st.dataframe(text_feature_display, use_container_width=True, hide_index=True)
        st.caption(
            "This table is stored at `data/processed/event_level.parquet` and `data/processed/event_level.csv`. "
            "It adds topic and severity fields to the event-level base, before market returns and CAR windows are attached."
        )

        st.subheader("Topic Summary")
        st.dataframe(pd.read_csv(topic_summary_path), use_container_width=True, hide_index=True)

        st.subheader("Severity Buckets")
        st.dataframe(pd.read_csv(severity_summary_path), use_container_width=True, hide_index=True)

        with st.expander("Topic Classifier Algorithm"):
            st.markdown(
                """
The classifier uses the validated V3 grouped rule-based regex design.

1. Use the SEC staff-letter text as the primary classification text.
2. Count regex pattern matches for substantive SEC issue areas.
3. Select the highest-scoring detailed label as `topic_detail`.
4. If no substantive pattern is found but routine disclosure language appears, assign `routine_disclosure`.
5. Map `topic_detail` into a smaller grouped `topic` label for analysis, visualization, and regression.

`topic_detail` is the audit label. `topic` is the analysis label used in the main results.

The predefined detailed topic patterns are:

| `topic_detail` | Pattern focus |
|---|---|
| `revenue` | revenue recognition, ASC 606, deferred revenue, performance obligations |
| `non_gaap` | non-GAAP, adjusted EBITDA, free cash flow, Regulation G |
| `mda_liquidity` | MD&A, liquidity, capital resources, cash flow, known trends |
| `segments` | segment reporting, operating segments, CODM, ASC 280 |
| `impairment_fair_value` | goodwill, impairment, fair value, valuation allowance |
| `business_combinations` | acquisition, merger, purchase price allocation, pro forma |
| `controls_accounting` | internal controls, ICFR, material weakness, restatement |
| `risk_legal_regulatory` | risk factors, litigation, cybersecurity, climate, regulatory |
| `tax_debt_equity` | income tax, debt, convertible securities, warrants, EPS |

The grouped analysis topics are:

| `topic_detail` inputs | Grouped `topic` |
|---|---|
| `controls_accounting`, `impairment_fair_value`, `tax_debt_equity` | `accounting_financial_reporting` |
| `revenue`, `non_gaap` | `revenue_non_gaap` |
| `mda_liquidity` | `mda_liquidity` |
| `segments`, `business_combinations` | `business_structure` |
| `risk_legal_regulatory` | `risk_legal_regulatory` |
| `routine_disclosure`, `other` | `routine_other` |
"""
            )

        with st.expander("Severity Scoring Algorithm"):
            st.markdown(
                """
`severity_score` is a continuous text-and-process intensity measure:

```text
severity_score =
    z(letter_word_count)
  + z(intensity_term_count)
  + z(thread_round_count)
  + amendment_indicator
```

- `letter_word_count`: longer staff letters indicate more extensive SEC comments.
- `intensity_term_count`: counts stronger request or concern language in the staff letter.
- `thread_round_count`: more staff-letter rounds indicate a longer review process.
- `amendment_indicator`: equals 1 when the thread text contains `amend`, `amended`, or `amendment`; otherwise 0. Amendment language is treated as a sign that the SEC review led to a disclosure revision or filing correction.

The score can be positive or negative because the first three inputs are z-scores, meaning each event is
measured relative to the sample average. A negative value means below-average severity in this sample,
not negative severity.

In the current baseline, `severity_score` ranges from about `-2.74` to `15.41`.

`severity_bucket` is a categorical version of the continuous score. The project uses `qcut` to split the
283 events into three roughly equal groups:

- `low`: lowest third, approximately `-2.74` to `-1.12`
- `medium`: middle third, approximately `-1.11` to `0.51`
- `high`: highest third, approximately `0.51` to `15.41`

The bucket is useful for presentation and distribution checks, while regressions mainly use the continuous
`severity_score`.
"""
            )


with tabs[5]:
    summary_path = WORKFLOW_DIR / "06_market_data_summary.json"
    coverage_path = TABLES_DIR / "market_coverage_summary.csv"
    full_market_csv_path = DATA_DIR / "processed" / "market_data.csv"
    full_market_parquet_path = DATA_DIR / "processed" / "market_data.parquet"

    st.header("Market Data")
    st.write(
        "This step prepares daily prices, volumes, and returns for event tickers plus the benchmark. "
        "These rows feed the later abnormal-return and CAR construction step."
    )

    if not summary_path.exists():
        st.warning("Step 6 artifacts are not available yet. Run `./.venv/bin/python run.py --step market`.")
    else:
        summary = load_json(summary_path)
        metrics = summary["metrics"]
        date_range = f"{metrics['start_date']} to {metrics['end_date']}"
        summary_table = pd.DataFrame(
            [
                {
                    "market_rows": metrics["row_count"],
                    "tickers": metrics["ticker_count"],
                    "event_tickers": metrics["event_ticker_count"],
                    "covered_event_tickers": metrics["event_ticker_covered_count"],
                    "missing_event_tickers": metrics["event_ticker_missing_count"],
                    "benchmark": metrics["benchmark_ticker"],
                    "date_range": date_range,
                }
            ]
        )

        st.subheader("Market Data Summary")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        full_market_df = read_table(full_market_csv_path, full_market_parquet_path)
        market_display = full_market_df.copy()

        market_tickers = sorted(market_display["ticker"].dropna().unique().tolist())
        selected_market_ticker = st.selectbox("Market ticker", ["All tickers"] + market_tickers)
        if selected_market_ticker != "All tickers":
            market_display = market_display[market_display["ticker"] == selected_market_ticker]

        ticker_type = st.selectbox("Ticker type", ["All", "Event tickers", "Benchmark"])
        if ticker_type == "Event tickers":
            market_display = market_display[market_display["is_event_ticker"]]
        elif ticker_type == "Benchmark":
            market_display = market_display[market_display["is_benchmark"]]

        market_display["date"] = pd.to_datetime(market_display["date"], errors="coerce")
        min_date = market_display["date"].min().date()
        max_date = market_display["date"].max().date()
        selected_date_range = st.date_input("Market date range", value=(min_date, max_date))
        if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            market_display = market_display[
                (market_display["date"].dt.date >= start_date)
                & (market_display["date"].dt.date <= end_date)
            ]

        st.subheader("Market Data Table")
        st.caption(f"Showing {len(market_display):,} of {len(full_market_df):,} ticker-date rows.")
        st.dataframe(market_display, use_container_width=True, hide_index=True)
        st.caption(
            "This table is stored at `data/processed/market_data.parquet` and `data/processed/market_data.csv`. "
            "It is a daily ticker-return panel used by Step 7 for abnormal returns, CAR windows, and pre-event volatility."
        )

        with st.expander("Ticker Coverage"):
            st.dataframe(pd.read_csv(coverage_path), use_container_width=True, hide_index=True)

        with st.expander("Notes"):
            st.markdown(
                f"""
**Data source.** Market prices are loaded from the official raw market cache under `data/raw/market/`.
If the cache does not cover the configured event or pre-event windows, the pipeline refreshes the missing
coverage with yfinance and writes the result back into the official raw cache.

**Ticker universe.** The market panel includes the 93 event tickers plus `{metrics['benchmark_ticker']}`,
for 94 tickers total. All event tickers are covered in the current baseline.

**Benchmark.** `{metrics['benchmark_ticker']}` is the benchmark used later to compute market-adjusted
abnormal returns.

**Scope.** This step prepares close prices, volume, and daily returns. Abnormal returns, CAR windows, and
pre-event volatility are calculated in Step 7, not here.
"""
            )


with tabs[6]:
    summary_path = WORKFLOW_DIR / "07_event_study_regression_summary.json"
    event_time_csv_path = DATA_DIR / "processed" / "event_time.csv"
    event_time_parquet_path = DATA_DIR / "processed" / "event_time.parquet"
    regression_csv_path = DATA_DIR / "processed" / "regression_dataset.csv"
    regression_parquet_path = DATA_DIR / "processed" / "regression_dataset.parquet"

    st.header("Event Study & Regression")
    st.write(
        "This step converts market returns into event-time abnormal returns, CAR windows, "
        "regression-ready rows, and baseline regression tables."
    )

    if not summary_path.exists():
        st.warning("Step 7 artifacts are not available yet. Run `./.venv/bin/python run.py --step event_study`.")
    else:
        summary = load_json(summary_path)
        metrics = summary["metrics"]
        event_time_df = read_table(event_time_csv_path, event_time_parquet_path)
        event_time_display = event_time_df.copy()

        event_time_tickers = sorted(event_time_display["ticker"].dropna().unique().tolist())
        selected_event_time_ticker = st.selectbox("Event-time ticker", ["All tickers"] + event_time_tickers)
        if selected_event_time_ticker != "All tickers":
            event_time_display = event_time_display[event_time_display["ticker"] == selected_event_time_ticker]

        event_ids = sorted(event_time_display["event_id"].dropna().unique().tolist())
        selected_event_id = st.selectbox("Event ID", ["All events"] + event_ids)
        if selected_event_id != "All events":
            event_time_display = event_time_display[event_time_display["event_id"] == selected_event_id]

        relative_day_min = int(event_time_display["relative_day"].min())
        relative_day_max = int(event_time_display["relative_day"].max())
        relative_day_range = st.slider(
            "Relative trading-day range",
            min_value=relative_day_min,
            max_value=relative_day_max,
            value=(relative_day_min, relative_day_max),
        )
        event_time_display = event_time_display[
            (event_time_display["relative_day"] >= relative_day_range[0])
            & (event_time_display["relative_day"] <= relative_day_range[1])
        ]

        st.subheader("Event-Time Dataset")
        st.caption(
            f"Showing {len(event_time_display):,} of {len(event_time_df):,} event-time rows. "
            "Each row is one SEC event by one relative trading day."
        )
        st.dataframe(event_time_display, use_container_width=True, hide_index=True)
        st.caption(
            "This table is stored at `data/processed/event_time.parquet` and `data/processed/event_time.csv`."
        )

        regression_df = read_table(regression_csv_path, regression_parquet_path)
        regression_cols = [
            "event_id",
            "ticker",
            "event_date",
            "industry",
            "topic",
            "topic_detail",
            "severity_score",
            "CAR_-3_-1",
            "CAR_-1_3",
            "CAR_1_60",
            "pre_event_volatility",
            "regression_inclusion_status",
        ]
        regression_display = regression_df[regression_cols].copy()

        regression_topics = sorted(regression_display["topic"].dropna().unique().tolist())
        selected_regression_topic = st.selectbox("Regression topic", ["All topics"] + regression_topics)
        if selected_regression_topic != "All topics":
            regression_display = regression_display[regression_display["topic"] == selected_regression_topic]

        inclusion_statuses = sorted(regression_display["regression_inclusion_status"].dropna().unique().tolist())
        selected_regression_status = st.selectbox("Regression inclusion status", ["All statuses"] + inclusion_statuses)
        if selected_regression_status != "All statuses":
            regression_display = regression_display[
                regression_display["regression_inclusion_status"] == selected_regression_status
            ]

        regression_ticker_query = st.text_input("Regression ticker search", "")
        if regression_ticker_query.strip():
            regression_display = regression_display[
                regression_display["ticker"].astype(str).str.contains(regression_ticker_query.strip(), case=False, na=False)
            ]

        st.subheader("Regression Dataset")
        st.caption(f"Showing {len(regression_display):,} of {len(regression_df):,} regression rows.")
        st.dataframe(regression_display, use_container_width=True, hide_index=True)
        st.caption(
            "This table is stored at `data/processed/regression_dataset.parquet` and "
            "`data/processed/regression_dataset.csv`."
        )

        with st.expander("Notes"):
            st.markdown(
                f"""
**Event-time construction.** `event_time` combines the event-level table with the market-data panel.
Each observation is one `event_id` by one `relative_day`. The current baseline has
`{metrics['event_time_row_count']:,}` event-time rows for `{metrics['event_time_event_count']}` matched events.

**Abnormal return and CAR.** For event *i* and relative trading day *t*, abnormal return is
`AR_i,t = R_i,t - R_SPY,t`. This is a market-adjusted return design, not a CAPM or Fama-French expected-return model.
Each CAR window is the sum of daily abnormal returns inside that relative-day interval:
`CAR_-3_-1`, `CAR_-1_3`, and `CAR_1_60`.

**Pre-event volatility.** `pre_event_volatility` is the stock-return standard deviation over relative trading
days `{metrics['pre_event_volatility_window']}`. This uses a longer, earlier, non-overlapping window, avoiding
the immediate event period.

**Regression dataset construction.** `regression_dataset` collapses the event-time panel back to one row per
SEC event. It keeps event-level variables such as `topic`, `topic_detail`, `severity_score`, and `industry`,
then adds CAR windows and `pre_event_volatility`.

**Regression inclusion.** The regression table has `{metrics['regression_row_count']}` rows, with
`{metrics['regression_included_count']}` included rows. Rows are excluded when required regression fields
such as CAR, topic, severity, or pre-event volatility are missing.
"""
            )

