import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Qiskit imports are optional; we wrap them to allow graceful fallback if qiskit
# isn't installed in the environment where this code is inspected.
try:
    from qiskit_aer import Aer  # noqa: F401
    from qiskit.circuit.library import ZZFeatureMap  # type: ignore
    from qiskit.primitives import StatevectorSampler  # type: ignore
    from qiskit_algorithms.state_fidelities import ComputeUncompute  # type: ignore
    from qiskit_machine_learning.kernels import FidelityQuantumKernel  # type: ignore
    _QISKIT_AVAILABLE = True
except Exception:
    _QISKIT_AVAILABLE = False

# Paths (use models/quantum_models as repository stores artifacts there)
BASE_MODELS = Path(__file__).resolve().parent / "models" / "quantum_models"
# Prefer datasets CSV in app/datasets
DF_PATH = Path(__file__).resolve().parent / "datasets" / "education_policies.csv"
VECTORIZER_PATH = BASE_MODELS / "tfidf_vectorizer.pkl"
X_TFIDF_PATH = BASE_MODELS / "X_tfidf.npy"

# --- Load dataset and model artifacts with fallbacks ---
def _load_df():
    try:
        if DF_PATH.exists():
            return pd.read_csv(DF_PATH)
        # Try to load education df from education models pickle if present
        edu_pkg = Path(__file__).resolve().parent / "models" / "education_models" / "education_policy_tfidf_matrix.pkl"
        if edu_pkg.exists():
            pkg = joblib.load(str(edu_pkg))
            df = pkg.get('df')
            if df is not None:
                return df
    except Exception:
        pass
    return pd.DataFrame()


def _load_vectorizer():
    try:
        if VECTORIZER_PATH.exists():
            return joblib.load(str(VECTORIZER_PATH))
    except Exception:
        pass
    return None


def _load_X_tfidf():
    try:
        if X_TFIDF_PATH.exists():
            # saved as numpy file or joblib; try both
            try:
                return np.load(str(X_TFIDF_PATH), allow_pickle=True)
            except Exception:
                return joblib.load(str(X_TFIDF_PATH))
    except Exception:
        pass
    return None


# load artifacts at module import time but gracefully
DF = _load_df()
VECTORIZER = _load_vectorizer()
X_TFIDF = _load_X_tfidf()

# --- Classical fallback (when Qiskit or artifacts are missing) ---
try:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

def _get_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return the real column name in df matching name (case-insensitive)."""
    lower = name.lower()
    for c in df.columns:
        if c.lower() == lower:
            return c
    return None

def _ensure_tfidf_from_df():
    """If VECTORIZER/X_TFIDF are missing and sklearn is present, build them from CSV."""
    global VECTORIZER, X_TFIDF, DF
    if not _SKLEARN_AVAILABLE:
        return
    if DF is None or DF.empty:
        return
    if VECTORIZER is not None and X_TFIDF is not None:
        return
    # Build corpus from common policy text fields
    fields = []
    for cand in ["title", "summary", "goals", "sector", "region"]:
        col = _get_column(DF, cand)
        if col:
            fields.append(col)
    if not fields:
        return
    text = DF[fields].astype(str).agg(" ".join, axis=1).fillna("")
    try:
        vec = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
        X = vec.fit_transform(text.values)
        VECTORIZER = vec
        X_TFIDF = X
    except Exception:
        # leave as-is on failure
        pass


def retrieve_top_k(query: str, k: int = 3) -> List[Tuple[Optional[str], Optional[str], float]]:
    """Search policies using a quantum kernel when available; otherwise
    fallback to classical cosine similarity with TF‑IDF. Returns list of
    (policy_id, title, similarity_score).
    """
    if DF is None or DF.empty:
        return []

    # Ensure we have a vectorizer/matrix
    if VECTORIZER is None or X_TFIDF is None:
        _ensure_tfidf_from_df()
    if VECTORIZER is None or X_TFIDF is None:
        return []

    try:
        # Transform query using vectorizer
        q_vec_sparse = VECTORIZER.transform([str(query)])

        if _QISKIT_AVAILABLE:
            try:
                # Convert to dense for Qiskit feature map dimension
                q_vec = q_vec_sparse.toarray()
                sampler = StatevectorSampler()
                feature_map = ZZFeatureMap(feature_dimension=q_vec.shape[1], reps=1)
                fidelity = ComputeUncompute(sampler=sampler)
                quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
                # If X_TFIDF is sparse, convert to dense array safely (could be large)
                X_dense = X_TFIDF.toarray() if hasattr(X_TFIDF, 'toarray') else X_TFIDF
                q_kernel = quantum_kernel.evaluate(q_vec, X_dense)[0]
                sims = q_kernel
            except Exception:
                # Fallback to classical similarity on failure
                sims = _cosine_similarity(q_vec_sparse, X_TFIDF).flatten() if _SKLEARN_AVAILABLE else np.array([])
        else:
            sims = _cosine_similarity(q_vec_sparse, X_TFIDF).flatten() if _SKLEARN_AVAILABLE else np.array([])

        if sims.size == 0:
            return []

        top_indices = np.argsort(sims)[::-1][:int(k)]
        pid_col = _get_column(DF, 'policy_id') or _get_column(DF, 'id')
        title_col = _get_column(DF, 'title') or _get_column(DF, 'name')

        results: List[Tuple[Optional[str], Optional[str], float]] = []
        for i in top_indices:
            if i < len(DF):
                row = DF.iloc[int(i)]
                pid = row.get(pid_col) if pid_col else None
                title = row.get(title_col) if title_col else None
            else:
                pid = int(i)
                title = 'Unknown'
            results.append((pid, title, float(sims[i])))
        return results
    except Exception:
        return []

def search_quantum_dict(query: str, top_k: int = 3) -> List[Dict]:
    """Convenience helper returning a list of dicts with optional metadata
    for use in APIs or UIs."""
    res = retrieve_top_k(query, top_k)
    out: List[Dict] = []
    # Precompute columns
    c_title = _get_column(DF, 'title')
    c_sector = _get_column(DF, 'sector')
    c_year = _get_column(DF, 'year')
    c_summary = _get_column(DF, 'summary')
    for pid, title, score in res:
        item: Dict = {"Index": None, "Similarity": float(score)}
        # Map pid to index if possible
        try:
            if _get_column(DF, 'policy_id') and pid is not None:
                idx_list = DF.index[DF[_get_column(DF, 'policy_id')] == pid].tolist()
                if idx_list:
                    item["Index"] = int(idx_list[0])
        except Exception:
            item["Index"] = None
        if title is not None:
            item["Title"] = str(title)
        try:
            # If we have an index, pull metadata; else skip
            idx = item.get("Index")
            if isinstance(idx, int) and 0 <= idx < len(DF):
                row = DF.iloc[idx]
                if c_sector: item["Sector"] = str(row.get(c_sector, ""))
                if c_year: item["Year"] = str(row.get(c_year, ""))
                if c_summary: item["Summary"] = str(row.get(c_summary, ""))
        except Exception:
            pass
        out.append(item)
    return out

def search_quantum(query: str, top_k: int = 3) -> List[Dict]:
    """Compatibility wrapper returning lowercase keys for index/similarity
    along with optional metadata used by some UIs."""
    data = search_quantum_dict(query, top_k)
    out: List[Dict] = []
    for d in data:
        out.append({
            "index": d.get("Index"),
            "similarity": d.get("Similarity"),
            # pass-through optional metadata
            "Title": d.get("Title"),
            "Sector": d.get("Sector"),
            "Year": d.get("Year"),
            "Summary": d.get("Summary"),
        })
    return out


def format_policy_bullets(results, df=None, query=""):
    """Return a markdown/bullet formatted string summarizing the results.
    If df is not provided, uses the module-level DF when available.
    """
    df_in = df if df is not None else DF
    if not results:
        return f'No relevant policies found for query "{query}".'
    bullets = [f'For the search query "{query}", the most relevant education policy documents are:\n']
    for doc_id, title, score in results:
        try:
            row = df_in[df_in['policy_id'] == doc_id].iloc[0]
            sector = row.get('sector', '')
            region = row.get('region', '')
            year = row.get('year', '')
            target_group = row.get('target_group', '')
            status = row.get('status', '')
            funding = row.get('funding_million_usd', '')
            summary = row.get('summary', '')
            goals = row.get('goals', '')
        except Exception:
            sector = region = year = target_group = status = funding = summary = goals = ''

        bullets.append(
            f"- **\"{title}\"** (Policy ID: {doc_id}, Similarity Score: {score:.4f})\n"
            f"  A {sector} policy in {region} ({year}), targeting {target_group}, status: {status}, funding: ${funding}M.\n"
            f"  **Summary:** {summary}\n"
            f"  **Goals:** {goals}\n"
        )
    return "\n".join(bullets)


if __name__ == "__main__":
    q = input("Enter your search query: ")
    k = 3
    results = retrieve_top_k(q, k)
    print()
    print(format_policy_bullets(results, DF, q))
