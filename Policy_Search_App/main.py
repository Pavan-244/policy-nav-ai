import os
import re
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import traceback

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
import warnings

# Suppress sklearn InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Optional SciPy sparse support
try:
    from scipy import sparse
    has_scipy = True
except ImportError:
    sparse = None
    has_scipy = False

# Quantum imports
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import QuantumCircuit

# Custom preprocessing
from preprocessing import preprocess

# ----------------- FastAPI setup -----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full traceback to server console for debugging
    tb = traceback.format_exc()
    print(f"Unhandled exception during request {request.method} {request.url}: {exc}\n{tb}")
    # Return a friendly error fragment so the frontend doesn't get an opaque 500
    return templates.TemplateResponse("partials/error.html", {"request": request, "message": "Internal server error. Check server logs for details."}, status_code=500)

# ----------------- Model configuration -----------------
MODEL_CONFIG = {
    "nlp1": {
        "vectorizer": "models/healthcare_vectorizer.pkl",
        "matrix": "models/healthcare_tfidf_matrix.pkl",
        "csv": "datasets/healthcare_dataset.csv", 
    },
    "nlp2": {
        "vectorizer": "models/policy_vectorizer.pkl",
        "matrix": "models/policy_tfidf_matrix.pkl",
        "csv": "datasets/education_policies.csv",
    },
    "nlp3": {
        "vectorizer": "models/financial_news_vectorizer.pkl",
        "matrix": "models/financial_news_tfidf_matrix.pkl",
        "csv": "datasets/financial_news_events.csv",
    },
    "qnlp": {
        "vectorizer": "models/quantum_policy_kernel.pkl",  # pickled quantum kernel
        "matrix": "models/quantum_policy_matrix.pkl",
        "csv": "datasets/education_policies.csv",
    },
}

# ----------------- Load all models -----------------
models: Dict[str, Dict[str, Any]] = {}
load_errors: Dict[str, str] = {}
print("Loading models and data...")

for key, paths in MODEL_CONFIG.items():
    try:
        vec = joblib.load(paths["vectorizer"]) if paths.get("vectorizer") else None
        mat = joblib.load(paths["matrix"]) if paths.get("matrix") else None
        df = pd.read_csv(paths["csv"]) if paths.get("csv") else None

        # Handle quantum kernel
        kernel_obj = None
        if key == "qnlp" and vec is not None and hasattr(vec, "evaluate"):
            kernel_obj = vec
            vec = None

        models[key] = {"vectorizer": vec, "matrix": mat, "df": df, "kernel": kernel_obj}
        inner_shape = None
        try:
            if isinstance(mat, dict):
                for k in ("matrix", "kernel_matrix", "tfidf", "tfidf_matrix", "mat", "X"):
                    if k in mat:
                        inner_shape = getattr(mat[k], 'shape', None)
                        break
            else:
                inner_shape = getattr(mat, 'shape', None)
        except Exception:
            inner_shape = getattr(mat, 'shape', None)
        print(f"✅ Loaded '{key}' | matrix shape: {inner_shape} | rows: {len(df) if df is not None else '?'}")
    except Exception as e:
        print(f"❌ Failed to load '{key}': {e}")
        load_errors[key] = str(e)

if not models:
    raise RuntimeError("No models were loaded. Check MODEL_CONFIG paths and files.")

# ----------------- Quantum Kernel -----------------
quantum_circuit = QuantumCircuit(2)
quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)

# ----------------- Helper functions -----------------
def get_model_resources(model_key: str) -> Tuple[Any, Any, pd.DataFrame, Any]:
    if model_key in load_errors:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' load failed: {load_errors[model_key]}")
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    m = models[model_key]
    vec, mat, df, kernel = m.get("vectorizer"), m.get("matrix"), m.get("df"), m.get("kernel")
    if df is None:
        raise HTTPException(status_code=500, detail=f"Dataframe missing for '{model_key}'")
    if vec is None and model_key != "qnlp":
        raise HTTPException(status_code=500, detail=f"Vectorizer missing for '{model_key}'")
    if mat is None and model_key != "qnlp":
        raise HTTPException(status_code=500, detail=f"TF-IDF matrix missing for '{model_key}'")
    
    return vec, mat, df, kernel


def _normalize_matrix(mat: Any):
    """Convert a loaded matrix artifact into a numeric matrix or sparse matrix
    usable by sklearn.metrics.pairwise.cosine_similarity.

    Supported inputs:
    - scipy sparse matrix (passed through)
    - numpy ndarray (passed through)
    - pandas DataFrame (converted to numpy values)
    - list/tuple of dicts (converted with DictVectorizer)
    - list/tuple of arrays (vstacked)
    """
    # If a dict-style artifact was saved (common pattern: {'matrix': <sparse>, 'df': <df>}),
    # try to extract the actual matrix under common keys.
    if isinstance(mat, dict):
        for key in ("matrix", "mat", "tfidf", "tfidf_matrix", "kernel_matrix", "X"):
            if key in mat:
                mat = mat[key]
                break

    # pass-through for sparse matrices
    try:
        from scipy import sparse as _sparse
        is_sparse = isinstance(mat, _sparse.spmatrix)
    except Exception:
        is_sparse = False

    if is_sparse:
        return mat

    if isinstance(mat, np.ndarray):
        return mat

    if isinstance(mat, pd.DataFrame):
        return mat.values

    if isinstance(mat, (list, tuple)):
        if len(mat) == 0:
            raise RuntimeError("Empty matrix artifact")
        first = mat[0]
        if isinstance(first, dict):
            dv = DictVectorizer(sparse=True)
            try:
                return dv.fit_transform(mat)
            except Exception as e:
                raise RuntimeError(f"Failed to convert list-of-dicts matrix via DictVectorizer: {e}")
        else:
            try:
                return np.vstack([np.asarray(r) for r in mat])
            except Exception as e:
                raise RuntimeError(f"Failed to stack matrix rows: {e}")

    # last resort: try to coerce to ndarray
    try:
        arr = np.asarray(mat)
        if arr.dtype == object:
            raise RuntimeError("Matrix contains non-numeric objects")
        return arr
    except Exception as e:
        raise RuntimeError(f"Unsupported TF-IDF matrix format: {e}")

# ----------------- Similarity search -----------------
def find_similar_policies_for_model(model_key: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    vec, mat, df, kernel_obj = get_model_resources(model_key)

    if model_key != "qnlp":
        query_vec = vec.transform([query])
        # ensure the stored matrix is a numeric array or sparse matrix
        try:
            mat_norm = _normalize_matrix(mat)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TF-IDF matrix format error: {e}")

        # convert to dense arrays for cosine computation (safe fallback)
        if hasattr(query_vec, "toarray"):
            q_arr = query_vec.toarray()
        else:
            q_arr = np.asarray(query_vec)

        if hasattr(mat_norm, "toarray"):
            mat_arr = mat_norm.toarray()
        else:
            mat_arr = np.asarray(mat_norm)

        similarities = cosine_similarity(q_arr, mat_arr).flatten()
        top_k = min(top_k, len(similarities))
        indices = similarities.argsort()[::-1][:top_k]
        top_df = df.iloc[indices].copy()
        top_df["similarity"] = similarities[indices]
        top_df["doc_id"] = top_df.index
        return top_df.to_dict("records")

    else:
        # Quantum NLP path
        try:
            query_df = pd.DataFrame({"title": ["query"], "full_text": [query], "stakeholders": ["All"]})
            query_df = preprocess(query_df)
            X_query = vec.transform(query_df.get("text_for_nlp", [query])).toarray()
            X_full = vec.transform(df.get("text_for_nlp", df.iloc[:, 0].astype(str))).toarray()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quantum preprocessing/transform failed: {e}")

        if X_query.size == 0 or X_full.size == 0:
            raise HTTPException(status_code=500, detail="Empty TF-IDF matrices for quantum similarity")

        # Safe normalization
        X_query_norm = np.pi * X_query / max(np.max(X_query), 1.0)
        X_full_norm = np.pi * X_full / max(np.max(X_full), 1.0)

        k = kernel_obj if kernel_obj else quantum_kernel
        sim = np.asarray(k.evaluate(X_query_norm, X_full_norm)).reshape(-1)  # flatten

        top_k = min(top_k, len(sim))
        indices = np.argsort(sim)[::-1][:top_k]
        top_df = df.iloc[indices].copy()
        top_df["similarity"] = sim[indices]
        top_df["doc_id"] = top_df.index
        return top_df.to_dict("records")

# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": list(models.keys())})


@app.get("/status")
async def status():
    """Return a JSON summary of which models loaded, resources present, and any load errors."""
    summary = {}
    for key, m in models.items():
        vec = m.get("vectorizer")
        mat = m.get("matrix")
        df = m.get("df")
        kernel = m.get("kernel")
        summary[key] = {
            "has_vectorizer": bool(vec),
            "has_matrix": bool(mat),
            "has_dataframe": bool(df),
            "has_kernel": bool(kernel),
            "matrix_shape": getattr(mat, "shape", None),
            "df_rows": len(df) if df is not None else None,
        }

    # include load errors if any
    return {"models": summary, "load_errors": load_errors}

# Example model pages
# Map logical model keys to template base names
TEMPLATE_MAP = {
    "nlp1": "health",
    "nlp2": "education",
    "nlp3": "financial",
    "qnlp": "quantum",
}


def _make_model_page(key: str):
    async def _handler(request: Request):
        template_base = TEMPLATE_MAP.get(key, key)
        template_name = f"{template_base}.html"
        # Verify template file exists using a stable path (avoid relying on templates.directory)
        base_dir = os.path.dirname(__file__)
        template_path = os.path.join(base_dir, "templates", template_name)
        if not os.path.exists(template_path):
            msg = f"Template file not found: {template_name} (looked at {template_path})"
            print(msg)
            return templates.TemplateResponse("partials/error.html", {"request": request, "message": msg}, status_code=500)

        try:
            return templates.TemplateResponse(template_name, {"request": request, "model_key": key})
        except Exception as e:
            # Log traceback to console for debugging and return an error page
            import traceback
            tb = traceback.format_exc()
            print(f"Template render error for {template_name}: {e}\n{tb}")
            # Return a generic error message to the user (avoid leaking full traceback)
            return templates.TemplateResponse(
                "partials/error.html",
                {"request": request, "message": "Internal server error rendering page. Check server logs for details."},
                status_code=500,
            )
    return _handler


for route, model_key in [("/health", "nlp1"), ("/education", "nlp2"), ("/financial", "nlp3"), ("/quantum", "qnlp")]:
    app.get(route, response_class=HTMLResponse)(_make_model_page(model_key))


@app.get("/visualize/{model_key}", response_class=HTMLResponse)
async def visualize_page(request: Request, model_key: str):
    # ensure model exists or report error
    if model_key not in models and model_key not in load_errors:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": f"Unknown model '{model_key}'"}, status_code=404)

    # verify template exists
    template_path = os.path.join(os.path.dirname(__file__), "templates", "visualize.html")
    if not os.path.exists(template_path):
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": "visualize.html template not found."}, status_code=500)

    return templates.TemplateResponse("visualize.html", {"request": request, "model_key": model_key})

# ----------------- Search endpoints -----------------
@app.post("/search/{model_key}", response_class=HTMLResponse)
async def search_model(request: Request, model_key: str, query: str = Form(...), top_k: int = Form(5)):
    # Basic input validation and clamping
    q = (query or "").strip()
    if not q:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": "Query cannot be empty."}, status_code=400)

    try:
        top_k = int(top_k)
    except Exception:
        top_k = 5
    top_k = max(1, min(top_k, 50))  # clamp to [1, 50]

    try:
        # Validate resources first to provide clearer error messages
        try:
            vec, mat, df, kernel_obj = get_model_resources(model_key)
        except HTTPException as he:
            return templates.TemplateResponse("partials/error.html", {"request": request, "message": he.detail}, status_code=he.status_code)

        # Run the model-specific transform/similarity with guarded error handling
        try:
            if model_key != "qnlp":
                # ensure vectorizer/matrix are usable
                if vec is None:
                    raise RuntimeError(f"Vectorizer for model '{model_key}' is not available")
                if mat is None:
                    raise RuntimeError(f"TF-IDF matrix for model '{model_key}' is not available")

                query_vec = vec.transform([q])
                try:
                    mat_norm = _normalize_matrix(mat)
                except Exception as e:
                    raise RuntimeError(f"TF-IDF matrix format error: {e}")

                if hasattr(query_vec, "toarray"):
                    q_arr = query_vec.toarray()
                else:
                    q_arr = np.asarray(query_vec)

                if hasattr(mat_norm, "toarray"):
                    mat_arr = mat_norm.toarray()
                else:
                    mat_arr = np.asarray(mat_norm)

                similarities = cosine_similarity(q_arr, mat_arr).flatten()
                top_k_eff = min(top_k, len(similarities))
                indices = similarities.argsort()[::-1][:top_k_eff]
                top_df = df.iloc[indices].copy()
                top_df["similarity"] = similarities[indices]
                top_df["doc_id"] = top_df.index
                results = top_df.to_dict("records")
            else:

                # Quantum path uses preprocess + kernel evaluation
                query_df = pd.DataFrame({"title": ["query"], "full_text": [q], "stakeholders": ["All"]})
                query_df = preprocess(query_df)
                # defensively extract the column used for text features
                try:
                    query_text = query_df.get("text_for_nlp", [q])
                except Exception:
                    query_text = [q]
                try:
                    full_text_col = df.get("text_for_nlp", df.iloc[:, 0].astype(str))
                except Exception:
                    full_text_col = df.iloc[:, 0].astype(str)

                # If a pretrained transformer isn't available, fall back to building a TF-IDF on the dataset
                if vec is None:
                    try:
                        tfidf = TfidfVectorizer(max_features=20000)
                        # fit on the full_text_col (ensure it's an iterable of strings)
                        corpus = list(map(str, full_text_col))
                        tfidf.fit(corpus)
                        X_full = tfidf.transform(corpus)
                        X_query = tfidf.transform([" ".join(map(str, query_text))])
                    except Exception as e:
                        raise RuntimeError(f"Failed to build fallback TF-IDF transformer for quantum model: {e}")
                else:
                    X_query = vec.transform(query_text)
                    X_full = vec.transform(full_text_col)

                # handle sparse/dense consistently
                if hasattr(X_query, "toarray"):
                    X_query_arr = X_query.toarray()
                else:
                    X_query_arr = np.asarray(X_query)

                if hasattr(X_full, "toarray"):
                    X_full_arr = X_full.toarray()
                else:
                    X_full_arr = np.asarray(X_full)

                if X_query_arr.size == 0 or X_full_arr.size == 0:
                    raise RuntimeError("Empty feature matrices for quantum similarity")

                # Determine expected kernel input dimensionality (try feature_map parameters or qubits)
                k = kernel_obj if kernel_obj else quantum_kernel
                expected_dim = None
                try:
                    if hasattr(k, "feature_map") and hasattr(k.feature_map, "num_parameters"):
                        expected_dim = int(k.feature_map.num_parameters)
                    elif hasattr(k, "feature_map") and hasattr(k.feature_map, "num_qubits"):
                        expected_dim = int(k.feature_map.num_qubits)
                except Exception:
                    expected_dim = None

                if expected_dim is None:
                    # heuristic fallback
                    expected_dim = min(8, X_full_arr.shape[1])

                # Align feature dimensionality: reduce with SVD if TF-IDF has more features, or pad with zeros if fewer
                n_features = X_full_arr.shape[1]
                if n_features > expected_dim:
                    svd = TruncatedSVD(n_components=expected_dim, random_state=42)
                    X_full_aligned = svd.fit_transform(X_full_arr)
                    X_query_aligned = svd.transform(X_query_arr)
                elif n_features < expected_dim:
                    pad_cols = expected_dim - n_features
                    X_full_aligned = np.pad(X_full_arr, ((0, 0), (0, pad_cols)), mode="constant", constant_values=0.0)
                    X_query_aligned = np.pad(X_query_arr, ((0, 0), (0, pad_cols)), mode="constant", constant_values=0.0)
                else:
                    X_full_aligned = X_full_arr
                    X_query_aligned = X_query_arr

                # Normalize into the range expected by the kernel
                X_query_norm = np.pi * X_query_aligned / max(np.max(X_query_aligned), 1.0)
                X_full_norm = np.pi * X_full_aligned / max(np.max(X_full_aligned), 1.0)
                sim = np.asarray(k.evaluate(X_query_norm, X_full_norm)).reshape(-1)
                top_k_eff = min(top_k, len(sim))
                indices = np.argsort(sim)[::-1][:top_k_eff]
                top_df = df.iloc[indices].copy()
                top_df["similarity"] = sim[indices]
                top_df["doc_id"] = top_df.index
                results = top_df.to_dict("records")

            return templates.TemplateResponse("partials/results.html", {"request": request, "results": results, "model": model_key})
        except HTTPException:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error while computing similarities for model={model_key}: {e}\n{tb}")
            return templates.TemplateResponse("partials/error.html", {"request": request, "message": f"Search processing error: {str(e)}"}, status_code=500)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during search_model (model={model_key}, top_k={top_k}, query_len={len(q)}): {e}\n{tb}")
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": "Internal server error during search. Check server logs for details."}, status_code=500)

@app.post("/search", response_class=HTMLResponse)
async def search_default(request: Request, query: str = Form(...), model: str = Form("nlp1"), top_k: int = Form(5)):
    # Basic input validation and clamping
    q = (query or "").strip()
    if not q:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": "Query cannot be empty."}, status_code=400)

    if model not in models and model not in load_errors:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": f"Unknown model '{model}'."}, status_code=400)

    try:
        top_k = int(top_k)
    except Exception:
        top_k = 5
    top_k = max(1, min(top_k, 50))  # clamp to [1, 50]

    try:
        try:
            vec, mat, df, kernel_obj = get_model_resources(model)
        except HTTPException as he:
            return templates.TemplateResponse("partials/error.html", {"request": request, "message": he.detail}, status_code=he.status_code)

        try:
            if model != "qnlp":
                query_vec = vec.transform([q])
                try:
                    mat_norm = _normalize_matrix(mat)
                except Exception as e:
                    raise RuntimeError(f"TF-IDF matrix format error: {e}")

                if hasattr(query_vec, "toarray"):
                    q_arr = query_vec.toarray()
                else:
                    q_arr = np.asarray(query_vec)

                if hasattr(mat_norm, "toarray"):
                    mat_arr = mat_norm.toarray()
                else:
                    mat_arr = np.asarray(mat_norm)

                similarities = cosine_similarity(q_arr, mat_arr).flatten()
                top_k_eff = min(top_k, len(similarities))
                indices = similarities.argsort()[::-1][:top_k_eff]
                top_df = df.iloc[indices].copy()
                top_df["similarity"] = similarities[indices]
                top_df["doc_id"] = top_df.index
                results = top_df.to_dict("records")
            else:
                query_df = pd.DataFrame({"title": ["query"], "full_text": [q], "stakeholders": ["All"]})
                query_df = preprocess(query_df)
                X_query = vec.transform(query_df.get("text_for_nlp", [q])).toarray()
                X_full = vec.transform(df.get("text_for_nlp", df.iloc[:, 0].astype(str))).toarray()
                if X_query.size == 0 or X_full.size == 0:
                    raise RuntimeError("Empty TF-IDF matrices for quantum similarity")
                X_query_norm = np.pi * X_query / max(np.max(X_query), 1.0)
                X_full_norm = np.pi * X_full / max(np.max(X_full), 1.0)
                k = kernel_obj if kernel_obj else quantum_kernel
                sim = np.asarray(k.evaluate(X_query_norm, X_full_norm)).reshape(-1)
                top_k_eff = min(top_k, len(sim))
                indices = np.argsort(sim)[::-1][:top_k_eff]
                top_df = df.iloc[indices].copy()
                top_df["similarity"] = sim[indices]
                top_df["doc_id"] = top_df.index
                results = top_df.to_dict("records")

            return templates.TemplateResponse("partials/results.html", {"request": request, "results": results, "model": model})
        except HTTPException:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error while computing similarities for model={model}: {e}\n{tb}")
            return templates.TemplateResponse("partials/error.html", {"request": request, "message": f"Search processing error: {str(e)}"}, status_code=500)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during search_default (model={model}, top_k={top_k}, query_len={len(q)}): {e}\n{tb}")
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": "Internal server error during search. Check server logs for details."}, status_code=500)

# ----------------- NLP/Visualization helpers -----------------
STOPWORDS = set([
    "a","an","and","the","of","in","on","for","to","is","are","was","were","be","been","being",
    "by","with","as","at","from","or","that","this","these","those","it","its","into","their","his","her",
    "if","but","not","no","yes","we","you","they","them","us","our","your","i","me","my","mine",
    "about","over","under","than","then","so","such","can","could","should","would","will","may","might",
    "do","does","did","done","have","has","had"
])
WORD_RE = re.compile(r"[A-Za-z']+")

def _tokenize(text: str, min_len: int = 3) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    return [w for w in WORD_RE.findall(text.lower()) if len(w) >= min_len and w not in STOPWORDS]

def _infer_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = []
    for name, dtype in df.dtypes.items():
        if str(name).lower() == "doc_id":
            continue
        ctype = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
        cols.append({"name": str(name), "type": ctype, "dtype": str(dtype)})
    return cols

# ----------------- Visualization endpoints -----------------
@app.get("/api/visualize/{model_key}/columns")
async def viz_columns(model_key: str):
    _, _, df, _ = get_model_resources(model_key)
    return {"model": model_key, "columns": _infer_columns(df)}

@app.get("/api/visualize/{model_key}/summary")
async def viz_summary(model_key: str, column: str, top: int = 10, chartType: str = "bar"):
    _, _, df, _ = get_model_resources(model_key)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

    series = df[column]
    payload: Dict[str, Any] = {"chartType": chartType, "title": f"{model_key}:{column}"}

    if pd.api.types.is_numeric_dtype(series):
        counts, edges = np.histogram(series.dropna(), bins=min(max(top, 5), 50))
        labels = [f"{round(edges[i],2)}–{round(edges[i+1],2)}" for i in range(len(edges)-1)]
        values = counts.tolist()
    else:
        vc = series.fillna("(null)").astype(str).value_counts().head(top)
        labels = vc.index.tolist()
        values = vc.values.tolist()

    rows = [{"label": l, "value": v} for l, v in zip(labels, values)]
    payload.update({"labels": labels, "values": values, "rows": rows})
    return payload

@app.get("/api/visualize/{model_key}/nlp/terms")
async def viz_nlp_terms(model_key: str, column: str, top: int = 50, min_len: int = 3, sample: int = 2000):
    _, _, df, _ = get_model_resources(model_key)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

    series = df[column].astype(str).fillna("")
    if len(series) > sample:
        series = series.sample(sample, random_state=42)

    counter: Counter = Counter()
    for txt in series:
        counter.update(_tokenize(txt, min_len=min_len))

    most = counter.most_common(top)
    return {"model": model_key, "column": column, "terms": [{"text": t, "value": int(c)} for t, c in most]}

@app.get("/api/visualize/{model_key}/nlp/cooc")
async def viz_nlp_cooc(model_key: str, column: str, top_terms: int = 30, min_len: int = 3, min_cooc: int = 2, sample: int = 2000):
    _, _, df, _ = get_model_resources(model_key)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

    series = df[column].astype(str).fillna("")
    if len(series) > sample:
        series = series.sample(sample, random_state=42)

    docs_tokens: List[List[str]] = []
    counter: Counter = Counter()
    for txt in series:
        toks = _tokenize(txt, min_len=min_len)
        docs_tokens.append(list(set(toks)))
        counter.update(toks)

    top_vocab = {t for t, _ in counter.most_common(top_terms)}

    pair_counts: Dict[tuple, int] = defaultdict(int)
    for toks in docs_tokens:
        vocab_terms = [t for t in toks if t in top_vocab]
        for i in range(len(vocab_terms)):
            for j in range(i + 1, len(vocab_terms)):
                a, b = sorted([vocab_terms[i], vocab_terms[j]])
                pair_counts[(a, b)] += 1

    nodes = [{"id": t, "label": t, "value": int(counter[t])} for t in top_vocab]
    links = [{"source": a, "target": b, "value": int(c)} for (a, b), c in pair_counts.items() if c >= min_cooc]
    return {"model": model_key, "column": column, "nodes": nodes, "links": links}
