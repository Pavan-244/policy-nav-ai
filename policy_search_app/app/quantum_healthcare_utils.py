import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Optional Qiskit imports
try:
    from qiskit_aer import Aer  # noqa: F401
    from qiskit.circuit.library import ZZFeatureMap  # type: ignore
    from qiskit.primitives import StatevectorSampler  # type: ignore
    from qiskit_algorithms.state_fidelities import ComputeUncompute  # type: ignore
    from qiskit_machine_learning.kernels import FidelityQuantumKernel  # type: ignore
    _QISKIT_AVAILABLE = True
except Exception:
    _QISKIT_AVAILABLE = False

# Optional sklearn imports
try:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# Paths
DF_PATH = Path(__file__).resolve().parent / "datasets" / "healthcare_dataset.csv"
VECTORIZER_PATHS = [
    Path(__file__).resolve().parent / "models" / "healthcare_models" / "healthcare_vectorizer.pkl",
    Path(__file__).resolve().parent / "models" / "healtcare_models" / "healthcare_vectorizer.pkl",
]
MATRIX_PATHS = [
    Path(__file__).resolve().parent / "models" / "healthcare_models" / "healthcare_tfidf_matrix.pkl",
    Path(__file__).resolve().parent / "models" / "healtcare_models" / "healthcare_tfidf_matrix.pkl",
]

# --- Load dataset and artifacts with fallbacks ---

def _first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_df() -> pd.DataFrame:
    try:
        if DF_PATH.exists():
            return pd.read_csv(DF_PATH)
    except Exception:
        pass
    # Fallback via models pickle
    try:
        m = _first(MATRIX_PATHS)
        if m:
            pkg = joblib.load(str(m))
            df = pkg.get("df")
            if df is not None:
                return df
    except Exception:
        pass
    return pd.DataFrame()


def _load_vectorizer_matrix():
    vec, X = None, None
    try:
        v = _first(VECTORIZER_PATHS)
        if v:
            vec = joblib.load(str(v))
    except Exception:
        vec = None
    try:
        m = _first(MATRIX_PATHS)
        if m:
            pkg = joblib.load(str(m))
            X = pkg.get("matrix")
    except Exception:
        X = None
    return vec, X


DF = _load_df()
VECTORIZER, X_TFIDF = _load_vectorizer_matrix()


def _ensure_tfidf_from_df():
    global VECTORIZER, X_TFIDF
    if not _SKLEARN_AVAILABLE:
        return
    if VECTORIZER is not None and X_TFIDF is not None:
        return
    if DF is None or DF.empty:
        return
    # Build corpus from healthcare descriptive columns
    def _col(df, name):
        lname = str(name).lower()
        for c in df.columns:
            if c.lower() == lname:
                return c
        return None
    fields = []
    for cand in ["Name", "Medical Condition", "Doctor", "Hospital", "Medication", "Test Results"]:
        c = _col(DF, cand)
        if c:
            fields.append(c)
    if not fields:
        return
    text = DF[fields].astype(str).agg(" ".join, axis=1).fillna("")
    try:
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = vec.fit_transform(text.values)
        VECTORIZER, X_TFIDF = vec, X
    except Exception:
        pass


def search_quantum_healthcare(query: str, top_k: int = 3) -> List[Dict]:
    """Quantum-style semantic search for healthcare. Falls back to classical TF-IDF
    cosine similarity when Qiskit artifacts are unavailable.
    Returns dicts with keys: Name, Medical Condition, Doctor, Hospital, Similarity.
    """
    if DF is None or DF.empty:
        return []
    if VECTORIZER is None or X_TFIDF is None:
        _ensure_tfidf_from_df()
    if VECTORIZER is None or X_TFIDF is None:
        return []

    try:
        q_vec_sparse = VECTORIZER.transform([str(query)])
        # Optional quantum kernel path
        if _QISKIT_AVAILABLE:
            try:
                qv = q_vec_sparse.toarray()
                feature_dim = qv.shape[1]
                sampler = StatevectorSampler()
                feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
                fidelity = ComputeUncompute(sampler=sampler)
                quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
                X_dense = X_TFIDF.toarray() if hasattr(X_TFIDF, 'toarray') else X_TFIDF
                sims = quantum_kernel.evaluate(qv, X_dense)[0]
            except Exception:
                sims = _cosine_similarity(q_vec_sparse, X_TFIDF).flatten() if _SKLEARN_AVAILABLE else np.array([])
        else:
            sims = _cosine_similarity(q_vec_sparse, X_TFIDF).flatten() if _SKLEARN_AVAILABLE else np.array([])
        if sims.size == 0:
            return []
        idx = np.argsort(sims)[::-1][:int(top_k)]
        out: List[Dict] = []
        for i in idx:
            if i >= len(DF):
                continue
            row = DF.iloc[int(i)]
            out.append({
                "Name": str(row.get(next((c for c in DF.columns if c.lower()=="name"), ""), "")),
                "Medical Condition": str(row.get(next((c for c in DF.columns if c.lower()=="medical condition"), ""), "")),
                "Doctor": str(row.get(next((c for c in DF.columns if c.lower()=="doctor"), ""), "")),
                "Hospital": str(row.get(next((c for c in DF.columns if c.lower()=="hospital"), ""), "")),
                "Similarity": float(sims[i])
            })
        return out
    except Exception:
        return []
