"""
BioTrial Finder - Retrieval & Ranking Demo App (Streamlit)
Run with:  streamlit run app_retrieval_demo.py

Uses the real dataset handed off by Person 1 (Prachi):
final_cleaned_trials.xls
"""

import streamlit as st
from retrieval_ranking import ClinicalTrialRetrieval

CSV_PATH = "final_cleaned_trials.csv"

st.set_page_config(page_title="BioTrial Finder - Retrieval Module", layout="wide")
st.title("🔎 BioTrial Finder — Retrieval & Ranking Module")
st.caption("Person 2's module: Indexing, Keyword Matching, TF-IDF, Cosine Similarity, Ranking, Similar Trial Finder")

engine = ClinicalTrialRetrieval(CSV_PATH)
st.success(f"Loaded {len(engine.df)} clinical trials.")

st.header("Search Clinical Trials")
query = st.text_input("Enter disease / drug / therapy type", "breast cancer immunotherapy")
top_n = st.slider("Number of results", 1, 20, 10)

if st.button("Search"):
    results = engine.search(query, top_n=top_n)
    if results.empty:
        st.warning("No matching trials found.")
    else:
        st.dataframe(results, use_container_width=True)

st.divider()

st.header("Find Similar Trials")
trial_ids = engine.df["NCT Number"].tolist()
selected_id = st.selectbox("Select a trial", trial_ids)
top_n_similar = st.slider("Number of similar trials", 1, 20, 10, key="similar_slider")

if st.button("Find Similar Trials"):
    similar = engine.find_similar_trials(selected_id, top_n=top_n_similar)
    st.dataframe(similar, use_container_width=True)
