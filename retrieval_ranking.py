"""
BioTrial Finder - Retrieval & Ranking Module (Person 2's work)
Now wired up to the REAL data handed off by Person 1 (Prachi):
  - final_cleaned_trials.xls  (4000 cleaned trials, already has a
    'combined_text' column ready for search)
  - final_bow_features.xls    (500-word bag-of-words matrix, same row
    order as final_cleaned_trials.xls - kept for reference / Person 3)

This module implements:
  - Indexing            (inverted index over combined_text)
  - Keyword Matching     (shortlist candidates from the index)
  - TF-IDF               (weight words by importance)
  - Cosine Similarity    (score candidate trials against the query)
  - Trial Ranking        (sort candidates by score)
  - Similar Trial Finder (given one trial, find the closest others)
"""

import re
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClinicalTrialRetrieval:
    def __init__(self, trials_csv: str, id_col: str = "NCT Number",
                 text_col: str = "combined_text"):
        # 1. Load Person 1's cleaned data and drop exact duplicate trials
        self.df = pd.read_csv(trials_csv)
        self.df = self.df.drop_duplicates(subset=id_col).reset_index(drop=True)

        self.id_col = id_col
        self.text_col = text_col
        self.df[self.text_col] = self.df[self.text_col].fillna("")

        # 2. TF-IDF vectorization of the already-cleaned combined_text
        #    (Person 1 did the cleaning/combining, we do the weighting +
        #    similarity math on top of it)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df[self.text_col])

        # 3. Build the inverted index for fast keyword shortlisting
        self.inverted_index = self._build_inverted_index()

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------
    def _build_inverted_index(self):
        """word -> set of row-numbers (trials) that contain that word."""
        index = defaultdict(set)
        for row_num, text in enumerate(self.df[self.text_col]):
            words = re.findall(r"\b\w+\b", text.lower())
            for w in set(words):
                index[w].add(row_num)
        return index

    # ------------------------------------------------------------------
    # KEYWORD MATCHING
    # ------------------------------------------------------------------
    def keyword_match(self, query: str):
        words = re.findall(r"\b\w+\b", query.lower())
        candidate_rows = set()
        for w in words:
            if w in self.inverted_index:
                candidate_rows |= self.inverted_index[w]
        return candidate_rows

    # ------------------------------------------------------------------
    # TRIAL RANKING (TF-IDF + Cosine Similarity)
    # ------------------------------------------------------------------
    def search(self, query: str, top_n: int = 10) -> pd.DataFrame:
        candidate_rows = self.keyword_match(query)
        if not candidate_rows:
            return pd.DataFrame(columns=[
                self.id_col, "Study Title", "Conditions", "Interventions",
                "primary_phase", "Study Status", "similarity_score"
            ])

        query_vector = self.vectorizer.transform([query.lower()])
        all_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        rows = sorted(candidate_rows, key=lambda r: all_scores[r], reverse=True)[:top_n]
        result = self.df.loc[rows].copy()
        result["similarity_score"] = [all_scores[r] for r in rows]

        cols = [self.id_col, "Study Title", "Conditions", "Interventions",
                "primary_phase", "Study Status", "similarity_score"]
        return result[cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # SIMILAR TRIAL FINDER
    # ------------------------------------------------------------------
    def find_similar_trials(self, trial_id: str, top_n: int = 10) -> pd.DataFrame:
        if trial_id not in self.df[self.id_col].values:
            return pd.DataFrame()

        row_num = self.df.index[self.df[self.id_col] == trial_id][0]
        scores = cosine_similarity(
            self.tfidf_matrix[row_num], self.tfidf_matrix
        ).flatten()

        result = self.df.copy()
        result["similarity_score"] = scores
        result = result[result[self.id_col] != trial_id]
        result = result.sort_values("similarity_score", ascending=False)

        cols = [self.id_col, "Study Title", "Conditions", "Interventions",
                "primary_phase", "Study Status", "similarity_score"]
        return result[cols].head(top_n).reset_index(drop=True)


# ----------------------------------------------------------------------
# DEMO - run this file directly to see it working on the real data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    engine = ClinicalTrialRetrieval("final_cleaned_trials.csv")

    print(f"Loaded {len(engine.df)} unique trials.\n")

    print("=== SEARCH: 'breast cancer immunotherapy' ===")
    print(engine.search("breast cancer immunotherapy", top_n=5)[
        ["NCT Number", "Study Title", "similarity_score"]
    ].to_string(), "\n")

    print("=== SEARCH: 'melanoma' ===")
    print(engine.search("melanoma", top_n=5)[
        ["NCT Number", "Study Title", "similarity_score"]
    ].to_string(), "\n")

    first_id = engine.df["NCT Number"].iloc[0]
    print(f"=== SIMILAR TRIALS TO {first_id} ===")
    print(engine.find_similar_trials(first_id, top_n=5)[
        ["NCT Number", "Study Title", "similarity_score"]
    ].to_string())
