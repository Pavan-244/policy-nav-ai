import joblib
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import re
from collections import Counter, defaultdict

# --- INITIALIZATION ---

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure model artifact filenames here.
# Adjust filenames/paths to match your saved artifacts.
MODEL_CONFIG = {
    # Health domain
    "nlp1": {
        "vectorizer": "models/healthcare_vectorizer.pkl",
        "matrix": "models/healthcare_tfidf_matrix.pkl",
    "csv": "datasets/healthcare_dataset.csv",  # used to build a DataFrame if matrix file doesn't include one
    },
    # Education domain
    "nlp2": {
        "vectorizer": "models/policy_vectorizer.pkl",
        "matrix": "models/policy_tfidf_matrix.pkl",
    "csv": "datasets/education_policies.csv",
    },
    # Financial domain
    "nlp3": {
        "vectorizer": "models/financial_news_vectorizer.pkl",
        "matrix": "models/financial_news_tfidf_matrix.pkl",
    "csv": "datasets/financial_news_events.csv",
    },
    # Quantum Education domain
    "qnlp": {
        "vectorizer": "models/quantum_tfidf_vectorizer.pkl",
        "matrix": "quantum_X_tfidf.npy",
    "csv": "datasets/education_policies.csv",
    },
}

# Load all model artifacts at startup into memory
models = {}
print("Loading models and data...")
for key, paths in MODEL_CONFIG.items():
    try:
        vec = joblib.load(paths["vectorizer"])  # type: ignore
        matrix_path = paths["matrix"]
        # Load matrix - joblib handles both .pkl and .npy (if joblib-pickled)
        data = joblib.load(matrix_path)  # type: ignore

        mat = None
        df = None

        # If packaged as dict with 'matrix' and 'df', use it directly
        if isinstance(data, dict) and "matrix" in data and "df" in data:
            mat = data["matrix"]
            df = data["df"]
        else:
            # Assume data is a sparse/dense matrix; try to attach a DataFrame
            mat = data
            csv_path = paths.get("csv")
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                except Exception as csv_err:
                    print(f"    ⚠️  Could not read CSV '{csv_path}' for model '{key}': {csv_err}")
            if df is None and hasattr(mat, "shape"):
                # Fallback: create a minimal DataFrame with only an index
                import numpy as np
                row_count = int(mat.shape[0])
                df = pd.DataFrame({"doc_id": np.arange(row_count)})
                print(f"    ℹ️  Using minimal DataFrame with {row_count} rows for model '{key}'.")

        models[key] = {"vectorizer": vec, "matrix": mat, "df": df}
        # Log shapes for debugging
        mat_shape = getattr(mat, "shape", "unknown")
        df_shape = len(df) if df is not None else 0
        print(f"  ✅ Loaded model '{key}': matrix_shape={mat_shape}, df_rows={df_shape}")
    except Exception as e:
        import traceback
        print(f"  ❌ Failed to load model '{key}': {e}")
        traceback.print_exc()

if not models:
    raise RuntimeError("No models were loaded. Check MODEL_CONFIG paths and files.")

# --- HELPER FUNCTIONS ---

def get_model_resources(model_key: str):
    """Return (vectorizer, tfidf_matrix, df) for model_key, raise 404 if missing."""
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    m = models[model_key]
    return m["vectorizer"], m["matrix"], m["df"]

def find_similar_policies_for_model(model_key: str, query: str, top_k: int = 5):
    """Find top_k similar policies for the given model key and query."""
    vectorizer, tfidf_matrix, full_df = get_model_resources(model_key)
    query_vec = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    results_df = full_df.iloc[top_indices].copy()
    # Ensure we carry an identifier for display purposes
    try:
        results_df["doc_id"] = results_df.index
    except Exception:
        # If index assignment fails, create a simple range id
        import numpy as np
        results_df["doc_id"] = np.arange(len(results_df))
    results_df["similarity"] = similarities[top_indices]
    return results_df.to_dict("records")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Main page. Your template can include a dropdown or buttons to choose between:
    nlp1, nlp2, nlp3, qnlp
    """
    # pass available model keys to the template so the UI can show selection options
    return templates.TemplateResponse(
        "index.html", {"request": request, "models": list(models.keys())}
    )


# Domain pages that load a pre-filled search form tied to a specific model key
@app.get("/health", response_class=HTMLResponse)
async def health_page(request: Request):
    return templates.TemplateResponse("health.html", {"request": request, "model_key": "nlp1"})


@app.get("/education", response_class=HTMLResponse)
async def education_page(request: Request):
    return templates.TemplateResponse("education.html", {"request": request, "model_key": "nlp2"})


@app.get("/financial", response_class=HTMLResponse)
async def financial_page(request: Request):
    return templates.TemplateResponse("financial.html", {"request": request, "model_key": "nlp3"})


@app.get("/quantum", response_class=HTMLResponse)
async def quantum_page(request: Request):
    return templates.TemplateResponse("quantum.html", {"request": request, "model_key": "qnlp"})

# Visualization page
@app.get("/visualize/{model_key}", response_class=HTMLResponse)
async def visualize_page(request: Request, model_key: str):
    # Validate model exists
    _ = get_model_resources(model_key)
    return templates.TemplateResponse("visualize.html", {"request": request, "model_key": model_key})

# Generic endpoint that accepts a model name in the path
@app.post("/search/{model_key}", response_class=HTMLResponse)
async def search_policies_model(request: Request, model_key: str, query: str = Form(...), top_k: int = Form(5)):
    """
    Search using a specific model.
    POST /search/{model_key} with form fields:
      - query: the text query
      - top_k: optional number of top results (default 5)
    """
    try:
        top_matches = find_similar_policies_for_model(model_key, query, top_k=top_k)
        return templates.TemplateResponse("partials/results.html", {"request": request, "results": top_matches, "model": model_key})
    except Exception as e:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": f"Search failed for model '{model_key}': {e}"}, status_code=500)

# Backwards-compatible endpoint: if form submits 'model' field to select model
@app.post("/search", response_class=HTMLResponse)
async def search_policies(request: Request, query: str = Form(...), model: str = Form("nlp1"), top_k: int = Form(5)):
    """
    Backwards-compatible POST /search form:
      - query: the text query
      - model: one of the keys from MODEL_CONFIG (defaults to 'nlp1')
      - top_k: optional number of top results
    """
    try:
        top_matches = find_similar_policies_for_model(model, query, top_k=top_k)
        return templates.TemplateResponse("partials/results.html", {"request": request, "results": top_matches, "model": model})
    except Exception as e:
        return templates.TemplateResponse("partials/error.html", {"request": request, "message": f"Search failed for model '{model}': {e}"}, status_code=500)


@app.get("/debug/models")
async def debug_models():
    lines = ["Loaded models summary:"]
    for k,m in models.items():
        try:
            shape = getattr(m["matrix"], "shape", None)
            n_rows = shape[0] if shape else "?"
            df_rows = len(m["df"]) if m.get("df") is not None else 0
            lines.append(f"- {k}: matrix_rows={n_rows}, df_rows={df_rows}")
        except Exception as e:
            lines.append(f"- {k}: error getting info: {e}")
    return PlainTextResponse("\n".join(lines))

# --- Visualization APIs ---

def _infer_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = []
    for name, dtype in df.dtypes.items():
        # Skip index-like columns if present
        if name.lower() in {"doc_id"}:
            continue
        kind = str(dtype)
        ctype = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
        cols.append({"name": name, "type": ctype, "dtype": kind})
    return cols

@app.get("/api/visualize/{model_key}/columns")
async def viz_columns(model_key: str):
    _, _, df = get_model_resources(model_key)
    return {"model": model_key, "columns": _infer_columns(df)}

@app.get("/api/visualize/{model_key}/summary")
async def viz_summary(model_key: str, column: str, top: int = 10, chartType: str = "bar"):
    _, _, df = get_model_resources(model_key)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

    series = df[column]
    payload: Dict[str, Any] = {"chartType": chartType, "title": f"{model_key}:{column}"}

    if pd.api.types.is_numeric_dtype(series):
        # Numeric: create bins using d3-like quantiles or pandas cut
        bins = min(max(top, 5), 50)
        try:
            counts, edges = np.histogram(series.dropna(), bins=bins)
            labels = [f"{round(edges[i],2)}–{round(edges[i+1],2)}" for i in range(len(edges)-1)]
            values = counts.tolist()
        except Exception:
            vc = series.dropna().value_counts().head(top)
            labels = vc.index.astype(str).tolist()
            values = vc.values.tolist()
        rows = [{"label": l, "value": v} for l, v in zip(labels, values)]
    else:
        vc = series.astype(str).fillna("(null)").value_counts().head(top)
        labels = vc.index.tolist()
        values = vc.values.tolist()
        rows = [{"label": l, "value": v} for l, v in zip(labels, values)]

    payload.update({"labels": labels, "values": values, "rows": rows})
    return payload

# --- NLP visualization endpoints: word cloud and related graph ---

# Minimal English stopwords set (extend as needed)
STOPWORDS = {
    "a","an","and","the","of","in","on","for","to","is","are","was","were","be","been","being",
    "by","with","as","at","from","or","that","this","these","those","it","its","into","their","his","her",
    "if","but","not","no","yes","we","you","they","them","us","our","your","i","me","my","mine",
    "about","over","under","than","then","so","such","can","could","should","would","will","may","might",
    "do","does","did","done","have","has","had"
}

WORD_RE = re.compile(r"[A-Za-z']+")

def _tokenize(text: str, min_len: int = 3) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    words = WORD_RE.findall(text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]

@app.get("/api/visualize/{model_key}/nlp/terms")
async def viz_nlp_terms(model_key: str, column: str, top: int = 50, min_len: int = 3, sample: int = 2000):
    _, _, df = get_model_resources(model_key)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    series = df[column].astype(str).fillna("")
    # Optional down-sampling for performance on large datasets
    if sample and len(series) > sample:
        series = series.sample(sample, random_state=42)
    counter: Counter[str] = Counter()
    for txt in series:
        counter.update(_tokenize(txt, min_len=min_len))
    most = counter.most_common(int(top))
    return {"model": model_key, "column": column, "terms": [{"text": t, "value": int(c)} for t,c in most]}

@app.get("/api/visualize/{model_key}/nlp/cooc")
async def viz_nlp_cooc(model_key: str, column: str, top_terms: int = 30, min_len: int = 3, min_cooc: int = 2, sample: int = 2000):
    _, _, df = get_model_resources(model_key)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    series = df[column].astype(str).fillna("")
    if sample and len(series) > sample:
        series = series.sample(sample, random_state=42)

    # Term frequency to select vocabulary
    counter: Counter[str] = Counter()
    docs_tokens: List[List[str]] = []
    for txt in series:
        toks = _tokenize(txt, min_len=min_len)
        docs_tokens.append(list(set(toks)))  # unique per doc to avoid over-counting
        counter.update(toks)
    top_vocab = {t for t,_ in counter.most_common(int(top_terms))}

    # Co-occurrence counts (document-level)
    pair_counts: Dict[tuple, int] = defaultdict(int)
    for toks in docs_tokens:
        vocab_terms = [t for t in toks if t in top_vocab]
        n = len(vocab_terms)
        for i in range(n):
            for j in range(i+1, n):
                a, b = vocab_terms[i], vocab_terms[j]
                if a > b:
                    a, b = b, a
                pair_counts[(a,b)] += 1

    nodes = [{"id": t, "label": t, "value": int(counter[t])} for t in top_vocab]
    links = [{"source": a, "target": b, "value": int(c)} for (a,b), c in pair_counts.items() if c >= int(min_cooc)]
    return {"model": model_key, "column": column, "nodes": nodes, "links": links}