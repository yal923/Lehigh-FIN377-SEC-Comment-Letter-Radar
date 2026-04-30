import streamlit as st


st.title("SEC Comment Letter Radar")
st.caption("Research design and high-level summary for the SEC comment-letter event-study project.")

st.header("Research Question")
st.write(
    "Do SEC comment letters contain regulatory signals that are reflected in stock returns, "
    "and do topic and severity measures explain variation in market reaction?"
)

st.header("Research Design")
st.markdown(
    """
1. Build a public-firm universe from S&P 500 constituents.
2. Preserve SEC comment-letter correspondence artifacts.
3. Convert filing-level correspondence into review threads.
4. Convert threads into event-level observations.
5. Classify each event by topic and compute a severity score.
6. Link each event to stock and benchmark market returns.
7. Estimate market-adjusted CAR windows and regression specifications.
"""
)

st.header("Main Outputs")
st.markdown(
    """
- **Background & Methods** explains each pipeline step and shows real intermediate tables.
- **Final Results** presents topic-level CAR patterns, the regression-ready analysis dataset, and regression summary tables.
"""
)

st.header("Current Baseline")
st.markdown(
    """
- Sample: 100-company, 10-year SEC comment-letter baseline.
- Abnormal return model: stock return minus SPY return.
- Main text variables: V3 grouped topic classifier and severity score.
- Main regression controls: industry and pre-event volatility over relative trading days `[-150,-31]`.
"""
)
