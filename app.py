from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="SEC Comment Letter Project", layout="wide")

st.title("SEC Comment Letter Dashboard")
st.caption("Regulatory text signals, event-study evidence, and presentation-ready workflow outputs.")

st.header("Current Build")
st.write(
    "The official project is being rebuilt from the verified V3 notebook workflow. "
    "The first implemented step creates the S&P 500 reference dataset and workflow artifacts."
)

st.header("Key Features")
st.markdown(
    """
1. Clear and transparent explanation of the methodology
   - Follow each data-processing step from source files to final tables.
   - Review real preview tables instead of placeholder data.
   - See the files created at each stage.

2. SEC comment-letter event-study results
   - Compare topic-level and event-level outcomes.
   - Inspect final regression and CAR outputs from the V3 prototype run.

3. Project handoff and reproducibility
   - Keep official code separate from the archived prototype.
   - Use CSV/JSON for human validation and Parquet for efficient pandas/Streamlit loading.
"""
)

st.header("Current Build")
cols = st.columns(3)
cols[0].metric("Pipeline steps implemented", "1")
cols[1].metric("Workflow artifacts", "Step 1")
cols[2].metric("Prototype archive", "Available")

st.header("Navigation")
st.markdown(
    """
- **Background & Methods:** Step-by-step workflow tabs with real preview data.
- **Final Results:** Current V3 result artifacts retained from the verified prototype.
- **Research Design:** Research question and empirical design.
"""
)

st.subheader("Repository Layout")
st.code(
    """run.py
code/
data/
outputs/
pages/
Archive Prototype/""",
    language="text",
)
