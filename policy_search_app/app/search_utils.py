import joblib
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

# Prefer correct folder name; support older misspelling as fallback
VECTORIZER_CANDIDATES = [
    Path("app/models/healthcare_models/healthcare_vectorizer.pkl"),
    Path("app/models/healtcare_models/healthcare_vectorizer.pkl"),
]
MATRIX_CANDIDATES = [
    Path("app/models/healthcare_models/healthcare_tfidf_matrix.pkl"),
    Path("app/models/healtcare_models/healthcare_tfidf_matrix.pkl"),
]

vectorizer = None
tfidf_matrix = None
full_df = None

def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

try:
    vpath = _first_existing(VECTORIZER_CANDIDATES)
    if vpath:
        vectorizer = joblib.load(vpath)
except Exception:
    vectorizer = None

try:
    mpath = _first_existing(MATRIX_CANDIDATES)
    if mpath:
        data = joblib.load(mpath)
        tfidf_matrix = data.get("matrix")
        full_df = data.get("df")
except Exception:
    tfidf_matrix = None
    full_df = None

def _ensure_from_csv():
    """If artifacts are missing, build TF-IDF from the CSV on the fly."""
    global vectorizer, tfidf_matrix, full_df
    if vectorizer is not None and tfidf_matrix is not None and full_df is not None:
        return
    # Try reading the healthcare CSV
    csv_path = Path("app/datasets/healthcare_dataset.csv")
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    # Build a text corpus from common fields
    def col(df_in, name) -> Optional[str]:
        lower = name.lower()
        for c in df_in.columns:
            if c.lower() == lower:
                return c
        return None
    fields = [
        col(df, "Name"),
        col(df, "Medical Condition"),
        col(df, "Doctor"),
        col(df, "Hospital"),
        col(df, "Medication"),
        col(df, "Test Results"),
    ]
    fields = [f for f in fields if f]
    if not fields:
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = df[fields].astype(str).agg(" ".join, axis=1).fillna("")
    try:
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = vec.fit_transform(text.values)
        vectorizer = vec
        tfidf_matrix = X
        full_df = df
    except Exception:
        pass

def search_records(query, top_k=3, condition=None, doctor=None, hospital=None):
    # Ensure we have artifacts
    if vectorizer is None or tfidf_matrix is None or full_df is None:
        _ensure_from_csv()
    if vectorizer is None or tfidf_matrix is None or full_df is None:
        return []

    query_vec = vectorizer.transform([str(query).lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df_copy = full_df.copy()
    df_copy["Similarity"] = sims
    # Optional filters (case-insensitive safe)
    def _contains(series: pd.Series, val: Optional[str]):
        if not val:
            return pd.Series([True] * len(series), index=series.index)
        return series.astype(str).str.contains(str(val), case=False, na=False)
    if condition and "Medical Condition" in df_copy.columns:
        df_copy = df_copy[_contains(df_copy["Medical Condition"], condition)]
    if doctor and "Doctor" in df_copy.columns:
        df_copy = df_copy[_contains(df_copy["Doctor"], doctor)]
    if hospital and "Hospital" in df_copy.columns:
        df_copy = df_copy[_contains(df_copy["Hospital"], hospital)]

    top_records = df_copy.sort_values(by="Similarity", ascending=False).head(int(top_k))
    results = []
    for _, row in top_records.iterrows():
        results.append({
            "Name": row.get("Name", ""),
            "Age": row.get("Age", ""),
            "Gender": row.get("Gender", ""),
            "Blood Type": row.get("Blood Type", ""),
            "Medical Condition": row.get("Medical Condition", ""),
            "Date of Admission": row.get("Date of Admission", ""),
            "Doctor": row.get("Doctor", ""),
            "Hospital": row.get("Hospital", ""),
            "Insurance Provider": row.get("Insurance Provider", ""),
            "Billing Amount": row.get("Billing Amount", ""),
            "Room Number": row.get("Room Number", ""),
            "Admission Type": row.get("Admission Type", ""),
            "Discharge Date": row.get("Discharge Date", ""),
            "Medication": row.get("Medication", ""),
            "Test Results": row.get("Test Results", ""),
            "Similarity": float(row.get("Similarity", 0.0))
        })
    return results
