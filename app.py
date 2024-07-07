import streamlit as st
from streamlit_lottie import st_lottie
import json

#title of the page
st.markdown('## :rainbow[Resume Semantic Searcher :mag_right:]', unsafe_allow_html=True)

#option for query
st.text("Would you like to upload the link of resumes?")
col1, col2 = st.columns((0.3,0.7))
with col1:
  yes = st.button("Yes, I want to upload")
  if yes:
    st.switch_page("pages/yes_page.py")
with col2:
  no = st.button("No, Query Existing Resume")
  if no:
    st.switch_page("pages/no_page.py")
