import streamlit as st


pages = [
    st.Page("pages/00_Overview_Research_Design.py", title="Overview / Research Design"),
    st.Page("pages/01_Background_and_Methods.py", title="Background and Methods"),
    st.Page("pages/02_Final_Results.py", title="Final Results"),
]

navigation = st.navigation(pages)
navigation.run()
