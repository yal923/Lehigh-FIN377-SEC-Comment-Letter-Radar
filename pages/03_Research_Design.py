import streamlit as st


st.set_page_config(page_title="Research Design", layout="wide")
st.title("Research Design")
st.caption("Research framing for the SEC comment-letter event-study project.")

st.header("Question")
st.write(
    "Do SEC comment letters contain regulatory signals that are reflected in stock returns, "
    "and do topic and severity measures explain variation in market reaction?"
)

st.header("Empirical Flow")
st.markdown(
    """
1. Construct a firm universe.
2. Download and preserve SEC correspondence artifacts.
3. Build comment-letter threads.
4. Convert threads into event-level observations.
5. Add topic and severity text features.
6. Link event observations to market data.
7. Estimate event-study windows and regression outputs.
"""
)
