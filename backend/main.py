from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import re
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import threading

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"

app = FastAPI()
# Mount static files at /static so templates can call `url_for('static', path=...)`
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configure templates to load from the frontend templates folder
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.on_event("startup")
async def startup_event():
    """Preload the health model in the background to avoid first-request delay."""
    import asyncio
    
    def preload_health_model():
        try:
            print("[Startup] Preloading health model (nlp1) in background...")
            get_model_resources("nlp1")
            print("[Startup] Health model preloaded successfully!")
        except Exception as e:
            print(f"[Startup] Warning: Failed to preload health model: {e}")
    
    # Run in background thread to not block startup
    asyncio.create_task(asyncio.to_thread(preload_health_model))

# ----------------- Model configuration and loading -----------------
# Map logical model keys to their artifacts and dataset CSVs.
# Artifacts are optional; if missing, we'll build TF-IDF from the dataset at startup or on first use.
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # Health (nlp1) - Large dataset, optimized settings
    "nlp1": {
        "name": "health",
        "dataset": str(BASE_DIR / "frontend" / "datasets" / "healthcare_dataset.csv"),
        "vectorizer": str(BASE_DIR / "backend" / "models" / "healthcare_vectorizer.pkl"),
        "matrix": str(BASE_DIR / "backend" / "models" / "healthcare_tfidf_matrix.pkl"),
        "max_features": 5000,  # Reduced for large dataset
        "ngram_range": (1, 1),  # Unigrams only for speed
        "sample_size": 10000,  # Limit to 10k rows for faster processing
    },
    # Education policies (nlp2)
    "nlp2": {
        "name": "education",
        "dataset": str(BASE_DIR / "frontend" / "datasets" / "education_policies.csv"),
        "vectorizer": str(BASE_DIR / "backend" / "models" / "policy_vectorizer.pkl"),
        "matrix": str(BASE_DIR / "backend" / "models" / "policy_tfidf_matrix.pkl"),
        "max_features": 10000,
        "ngram_range": (1, 2),
    },
    # Financial news (nlp3)
    "nlp3": {
        "name": "financial",
        "dataset": str(BASE_DIR / "frontend" / "datasets" / "financial_news_events.csv"),
        "vectorizer": str(BASE_DIR / "backend" / "models" / "financial_news_vectorizer.pkl"),
        "matrix": str(BASE_DIR / "backend" / "models" / "financial_news_tfidf_matrix.pkl"),
        "max_features": 10000,
        "ngram_range": (1, 2),
    },
    # Quantum path: uses education dataset with quantum-trained vectorizer and matrix
    "qnlp": {
        "name": "quantum-education",
        "dataset": str(BASE_DIR / "frontend" / "datasets" / "education_policies.csv"),
        "vectorizer": str(BASE_DIR / "backend" / "models" / "quantum_policy_vectorizer.pkl"),
        "matrix": str(BASE_DIR / "backend" / "models" / "quantum_policy_tfidf_matrix.pkl"),
        "max_features": 10000,
        "ngram_range": (1, 2),
    },
}

# In-memory cache of loaded models and any load errors
models: Dict[str, Dict[str, Any]] = {}
load_errors: Dict[str, str] = {}
_load_lock = threading.Lock()

def _safe_load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle '{path}': {e}")

def _normalize_matrix(mat_any: Any):
    """Return a matrix-like that we can use with cosine_similarity / toarray().
    Accepts numpy arrays, scipy sparse matrices, pandas DataFrames, or dicts with 'matrix'.
    """
    m = mat_any
    if isinstance(m, dict) and "matrix" in m:
        m = m["matrix"]
    if isinstance(m, pd.DataFrame):
        m = m.values
    if sp.issparse(m):
        return m  # keep sparse
    if hasattr(m, "toarray"):
        return m
    try:
        return np.asarray(m)
    except Exception:
        raise RuntimeError("Unsupported matrix format; expected array-like/sparse/DataFrame/dict with 'matrix'.")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'text_for_nlp' column by concatenating reasonable text fields.
    Strategy: if common text columns exist, prefer them; otherwise join all object dtype columns.
    """
    df = df.copy()
    
    # Return early if already preprocessed
    if "text_for_nlp" in df.columns and df["text_for_nlp"].notna().any():
        return df
    
    # Common text fields for different datasets
    text_cols_priority = [
        "title", "summary", "goals", "full_text", "stakeholders",
        # financial dataset
        "Headline", "Market_Event", "Sector", "Related_Company", "Sentiment",
        # healthcare dataset  
        "Name", "Medical Condition", "Doctor", "Hospital", "Medication", "Test Results", "Admission Type",
    ]
    existing = [c for c in text_cols_priority if c in df.columns]
    if existing:
        parts = [df[c].fillna("").astype(str) for c in existing]
        df["text_for_nlp"] = (parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=". "))
    else:
        obj_cols = [c for c, t in df.dtypes.items() if t == "object"]
        if not obj_cols:
            # fallback to first few columns
            cols = df.columns[:5].tolist()
            obj_cols = cols
        parts = [df[c].fillna("").astype(str) for c in obj_cols]
        df["text_for_nlp"] = (parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=". "))
    
    df["text_for_nlp"] = df["text_for_nlp"].fillna("").astype(str)
    return df

class CosineQuantumKernel:
    """Simple kernel with an evaluate(X, Y) API using cosine similarity on normalized vectors."""
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Normalize rows to unit length for cosine
        def _row_norm(a):
            denom = np.linalg.norm(a, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            return a / denom
        Xn = _row_norm(X)
        Yn = _row_norm(Y)
        return Xn @ Yn.T

quantum_kernel = CosineQuantumKernel()

def _safe_load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle '{path}': {e}")

def _normalize_matrix(mat_any: Any):
    """Return a matrix-like that we can use with cosine_similarity / toarray().
    Accepts numpy arrays, scipy sparse matrices, pandas DataFrames, or dicts with 'matrix'.
    """
    m = mat_any
    if isinstance(m, dict) and "matrix" in m:
        m = m["matrix"]
    if isinstance(m, pd.DataFrame):
        m = m.values
    if sp.issparse(m):
        return m  # keep sparse
    if hasattr(m, "toarray"):
        return m
    try:
        return np.asarray(m)
    except Exception:
        raise RuntimeError("Unsupported matrix format; expected array-like/sparse/DataFrame/dict with 'matrix'.")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'text_for_nlp' column by concatenating reasonable text fields.
    Strategy: if common text columns exist, prefer them; otherwise join all object dtype columns.
    """
    df = df.copy()
    # Common text fields for education dataset
    text_cols_priority = [
        "title", "summary", "goals", "full_text", "stakeholders",
        # financial dataset
        "Headline", "Market_Event", "Sector", "Related_Company", "Sentiment",
        # healthcare dataset
        "Name", "Medical Condition", "Doctor", "Hospital", "Medication", "Test Results", "Admission Type",
    ]
    existing = [c for c in text_cols_priority if c in df.columns]
    if existing:
        parts = [df[c].astype(str) for c in existing]
        df["text_for_nlp"] = (parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=". "))
    else:
        obj_cols = [c for c, t in df.dtypes.items() if t == "object"]
        if not obj_cols:
            # fallback to first few columns
            cols = df.columns[:5].tolist()
            obj_cols = cols
        parts = [df[c].astype(str) for c in obj_cols]
        df["text_for_nlp"] = (parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=". "))
    return df

class CosineQuantumKernel:
    """Simple kernel with an evaluate(X, Y) API using cosine similarity on normalized vectors."""
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Normalize rows to unit length for cosine
        def _row_norm(a):
            denom = np.linalg.norm(a, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            return a / denom
        Xn = _row_norm(X)
        Yn = _row_norm(Y)
        return Xn @ Yn.T

quantum_kernel = CosineQuantumKernel()

def _build_tfidf(df: pd.DataFrame, model_key: str = None):
    """Build a TF-IDF vectorizer and matrix from dataframe using preprocess().
    For large datasets, applies sampling and optimized settings.
    """
    df2 = preprocess(df)
    
    # Get model-specific settings
    cfg = MODEL_CONFIG.get(model_key, {}) if model_key else {}
    max_features = cfg.get("max_features", 10000)
    ngram_range = cfg.get("ngram_range", (1, 2))
    sample_size = cfg.get("sample_size")
    
    # Sample large datasets for faster processing
    if sample_size and len(df2) > sample_size:
        print(f"[{model_key}] Dataset has {len(df2)} rows. Sampling {sample_size} rows for faster processing...")
        df2 = df2.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"[{model_key}] Building TF-IDF with max_features={max_features}, ngram_range={ngram_range}, rows={len(df2)}...")
    
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )
    X = vec.fit_transform(df2["text_for_nlp"].astype(str))
    
    print(f"[{model_key}] TF-IDF matrix shape: {X.shape}, sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])):.2%}")
    
    return vec, X, df2

def _load_model(model_key: str) -> Dict[str, Any]:
    cfg = MODEL_CONFIG.get(model_key)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Unknown model key '{model_key}'.")

    # Load dataset
    try:
        df = pd.read_csv(cfg["dataset"])  # rely on pandas detection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset for {model_key}: {e}")

    vec = None
    mat = None

    # Try loading pickled artifacts if present
    try:
        vpath = cfg.get("vectorizer")
        if vpath and Path(vpath).exists():
            vec = _safe_load_pickle(vpath)
    except Exception as e:
        load_errors[model_key] = f"Vectorizer load failed: {e}"

    try:
        mpath = cfg.get("matrix")
        if mpath and Path(mpath).exists():
            mat = _safe_load_pickle(mpath)
    except Exception as e:
        prev = load_errors.get(model_key, "")
        load_errors[model_key] = (prev + " | " if prev else "") + f"Matrix load failed: {e}"

    # Build from dataset if missing vectorizer or matrix
    if vec is None or mat is None:
        try:
            vec, mat, df = _build_tfidf(df, model_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build TF-IDF for {model_key}: {e}")
    else:
        # Ensure df has text_for_nlp for quantum path or future use
        try:
            df = preprocess(df)
        except Exception:
            pass

    return {"vectorizer": vec, "matrix": mat, "dataframe": df, "kernel": None}

def get_model_resources(model_key: str):
    # Cache load
    if model_key in models:
        m = models[model_key]
        return m["vectorizer"], m.get("matrix"), m["dataframe"], m.get("kernel")

    try:
        m = _load_model(model_key)
        models[model_key] = m
        return m["vectorizer"], m.get("matrix"), m["dataframe"], m.get("kernel")
    except HTTPException as he:
        load_errors[model_key] = he.detail
        raise
    except Exception as e:
        load_errors[model_key] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# Basic page routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the landing page index.html from frontend/templates."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/education", response_class=HTMLResponse)
async def education_page(request: Request):
    # Templates expect `model_key` for hx-post and visualize links
    return templates.TemplateResponse("education.html", {"request": request, "model_key": "nlp2"})


@app.get("/financial", response_class=HTMLResponse)
async def financial_page(request: Request):
    return templates.TemplateResponse("financial.html", {"request": request, "model_key": "nlp3"})


@app.get("/health", response_class=HTMLResponse)
async def health_page(request: Request):
    return templates.TemplateResponse("health.html", {"request": request, "model_key": "nlp1"})


@app.get("/quantum", response_class=HTMLResponse)
async def quantum_page(request: Request):
    return templates.TemplateResponse("quantum.html", {"request": request, "model_key": "qnlp"})


@app.get("/visualize/{model_key}", response_class=HTMLResponse)
async def visualize_page(request: Request, model_key: str):
    return templates.TemplateResponse("visualize.html", {"request": request, "model_key": model_key})


@app.post("/search", response_class=HTMLResponse)
async def search_default(request: Request, query: str = Form(...), model: str = Form("nlp1"), top_k: int = Form(5)):
    # Basic input validation and clamping
    q = (query or "").strip()
    if not q:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": "Query cannot be empty."}, status_code=400)

    # Validate against configured models, not just loaded cache
    if model not in MODEL_CONFIG:
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
                # Replace NaN values with None for JSON serialization
                results = top_df.fillna('').to_dict("records")
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
                # Replace NaN values with empty string for JSON serialization
                results = top_df.fillna('').to_dict("records")

            # Use model-specific result template
            template_map = {
                "nlp1": "partials/results_health.html",
                "nlp2": "partials/results_education.html",
                "nlp3": "partials/results_financial.html",
                "qnlp": "partials/results_quantum.html"
            }
            result_template = template_map.get(model, "partials/results.html")
            
            return templates.TemplateResponse(result_template, {"request": request, "results": results, "model": model})
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

# Support path-style search used by templates: /search/{model_key}
@app.post("/search/{model_key}", response_class=HTMLResponse)
async def search_with_model_path(request: Request, model_key: str, query: str = Form(...), top_k: int = Form(5)):
    # Delegate to the same logic as search_default by passing model via form param
    return await search_default(request, query=query, model=model_key, top_k=top_k)

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