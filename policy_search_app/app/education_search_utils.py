from typing import List, Optional, Dict
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Paths to new Education Policy model artifacts
EDU_POLICY_VECTORIZER_PATH = "app/models/education_policy_vectorizer.pkl"
EDU_POLICY_MATRIX_PATH = "app/models/education_policy_tfidf_matrix.pkl"

# Load model artifacts at import time with graceful fallback
try:
    _edu_vectorizer = joblib.load(EDU_POLICY_VECTORIZER_PATH)
except Exception:
    _edu_vectorizer = None

try:
    _edu_data = joblib.load(EDU_POLICY_MATRIX_PATH)
    _edu_tfidf_matrix = _edu_data.get("matrix")
    _edu_df = _edu_data.get("df")
except Exception:
    _edu_tfidf_matrix = None
    _edu_df = None

def search_policies(
    query: str,
    top_k: int = 3,
    sector: Optional[str] = None,
    region: Optional[str] = None,
    status: Optional[str] = None,
    year: Optional[str] = None,
) -> List[Dict]:
    """
    Semantic search over education policy dataset using TF-IDF & cosine similarity.
    Filters: sector, region, status, year.

    Returns a list of dicts with keys:
    ["Policy ID", "Title", "Sector", "Region", "Year", "Target Group", "Status",
     "Funding (USD M)", "Impact Score", "Stakeholders", "Summary", "Goals", "Similarity"]
    """
    if not _edu_vectorizer or _edu_tfidf_matrix is None or _edu_df is None or _edu_df.empty:
        return []

    try:
        query_vec = _edu_vectorizer.transform([str(query).lower()])
        sims = cosine_similarity(query_vec, _edu_tfidf_matrix).flatten()
        df_copy = _edu_df.copy()
        df_copy["similarity"] = sims

        # Optional filters (case-insensitive contains for text; exact match for year)
        if sector and "sector" in (c.lower() for c in df_copy.columns):
            # Find matching column case
            col = next((c for c in df_copy.columns if c.lower() == "sector"), None)
            if col:
                df_copy = df_copy[df_copy[col].astype(str).str.contains(sector, case=False, na=False)]
        if region and "region" in (c.lower() for c in df_copy.columns):
            col = next((c for c in df_copy.columns if c.lower() == "region"), None)
            if col:
                df_copy = df_copy[df_copy[col].astype(str).str.contains(region, case=False, na=False)]
        if status and "status" in (c.lower() for c in df_copy.columns):
            col = next((c for c in df_copy.columns if c.lower() == "status"), None)
            if col:
                df_copy = df_copy[df_copy[col].astype(str).str.contains(status, case=False, na=False)]
        if year and "year" in (c.lower() for c in df_copy.columns):
            col = next((c for c in df_copy.columns if c.lower() == "year"), None)
            if col:
                df_copy = df_copy[df_copy[col].astype(str) == str(year)]

        top_records = df_copy.sort_values("similarity", ascending=False).head(int(top_k))

        results: List[Dict] = []
        for _, row in top_records.iterrows():
            results.append({
                "Policy ID": row.get("policy_id", ""),
                "Title": row.get("title", ""),
                "Sector": row.get("sector", ""),
                "Region": row.get("region", ""),
                "Year": row.get("year", ""),
                "Target Group": row.get("target_group", ""),
                "Status": row.get("status", ""),
                "Funding (USD M)": row.get("funding_million_usd", ""),
                "Impact Score": row.get("impact_score", ""),
                "Stakeholders": row.get("stakeholders", ""),
                "Summary": row.get("summary", ""),
                "Goals": row.get("goals", ""),
                "Similarity": float(row.get("similarity", 0.0)),
            })
        return results
    except Exception:
        return []

def search_records_education(
    query: str,
    top_k: int = 3,
    program: Optional[str] = None,
    institution: Optional[str] = None,
    gender: Optional[str] = None,
) -> List[Dict]:
    """
    Backward-compatible wrapper for previous function signature used by main.py.
    Maps old filters to new ones where possible:
      program -> sector, institution -> region. Gender is ignored.
    """
    return search_policies(
        query=query,
        top_k=top_k,
        sector=program,
        region=institution,
        status=None,
        year=None,
    )

