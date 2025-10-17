@app.get("/healthcare/quantum/visualizations", response_class=HTMLResponse)
def healthcare_quantum_visualizations(request: Request):
    return templates.TemplateResponse("healthcare_quantum_visualizations.html", {"request": request})
# ------------------ Healthcare Quantum Visualizations API ------------------

from app.quantum_healthcare_utils import DF as hc_quantum_df

@app.get("/healthcare/quantum/api/chart-data")
def healthcare_quantum_chart_data(
    x_field: str = Query(..., description="X-axis field (categorical)"),
    y_metric: str = Query("count", description="Y-axis metric: count | sum | avg"),
    value_field: Optional[str] = Query(None, description="Numeric field for sum/avg"),
    limit: int = Query(10, description="Number of top items to show")
):
    df = hc_quantum_df
    if df is None or df.empty or not x_field:
        return {"labels": [], "data": []}
    def col_for(df_in, name):
        lower = name.lower()
        for c in df_in.columns:
            if c.lower() == lower:
                return c
        return None
    x_col = col_for(df, x_field)
    if not x_col:
        return {"labels": [], "data": []}
    y_metric = (y_metric or "count").lower()
    if y_metric == "count":
        counts = df[x_col].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    else:
        val_col = col_for(df, value_field or "")
        if not val_col:
            return {"labels": [], "data": []}
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        grouped = df.groupby(x_col)[val_col]
        agg = grouped.sum() if y_metric == "sum" else grouped.mean()
        agg = agg.sort_values(ascending=False).head(limit)
        return {"labels": agg.index.astype(str).tolist(), "data": agg.values.tolist()}

@app.get("/healthcare/quantum/api/fields")
def healthcare_quantum_fields():
    df = hc_quantum_df
    if df is None or df.empty:
        return {"categorical": [], "numeric": []}
    categorical = []
    numeric = []
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(str(c))
        else:
            categorical.append(str(c))
    def _dedupe(seq):
        seen = set(); out = []
        for x in seq:
            lx = x.lower()
            if lx in seen: continue
            seen.add(lx); out.append(x)
        return out
    return {"categorical": _dedupe(categorical), "numeric": _dedupe(numeric)}
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
import pandas as pd
import joblib
import os
# Robust imports for local/package execution
try:
    from app.search_utils import search_records
except Exception:
    from search_utils import search_records  # type: ignore

try:
    from app.education_search_utils import search_records_education, search_policies
except Exception:
    from education_search_utils import search_records_education, search_policies  # type: ignore
try:
    from app.quantum_search_utils import search_quantum
except Exception:
    try:
        from quantum_search_utils import search_quantum  # type: ignore
    except Exception:
        from app.education_search_utils import search_quantum  # type: ignore

# Financial news search utils
try:
    from app.financial_search_utils import search_financial_news
except Exception:
    from financial_search_utils import search_financial_news  # type: ignore

# Healthcare quantum search utils (separate)
try:
    from app.quantum_healthcare_utils import search_quantum_healthcare
except Exception:
    try:
        from quantum_healthcare_utils import search_quantum_healthcare  # type: ignore
    except Exception:
        search_quantum_healthcare = None  # type: ignore
import pandas as pd
import joblib

app = FastAPI(
    title="PolicyNav - Public Policy Navigation Using AI",
    description="AI-Powered Policy Search & Analysis API using NLP, TF-IDF & Cosine Similarity",
    version="1.0"
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load dataset for visualizations (Healthcare) with CSV fallback
try:
    # Try correct folder name first
    data = joblib.load("app/models/healthcare_models/healthcare_tfidf_matrix.pkl")
    df = data.get("df")
except Exception:
    # Fallback to older/misspelled folder name used in some repos
    try:
        data = joblib.load("app/models/healtcare_models/healthcare_tfidf_matrix.pkl")
        df = data.get("df")
    except Exception:
        data = None
        df = None

if df is None or getattr(df, 'empty', True):
    try:
        df = pd.read_csv("app/datasets/healthcare_dataset.csv")
    except Exception:
        df = pd.DataFrame()

## Load dataset for Financial News visualizations (graceful fallback if missing)
try:
    fin_data = joblib.load("app/models/financial_models/financial_news_tfidf_matrix.pkl")
    fin_df = fin_data.get("df")
except Exception:
    fin_data = None
    fin_df = None

# CSV fallback for financial dataset
if fin_df is None or getattr(fin_df, 'empty', True):
    try:
        fin_df = pd.read_csv("app/datasets/financial_news_events.csv")
    except Exception:
        fin_df = pd.DataFrame()

# ---- Helpers ----
def _fields_for_df(in_df: pd.DataFrame):
    try:
        if in_df is None or in_df.empty:
            return {"categorical": [], "numeric": []}
        categorical = []
        numeric = []
        for c in list(in_df.columns):
            s = in_df[c]
            # Treat numeric dtypes as numeric; everything else as categorical
            if pd.api.types.is_numeric_dtype(s):
                numeric.append(str(c))
            else:
                categorical.append(str(c))
        # de-dup while preserving order
        def _dedupe(seq):
            seen = set(); out = []
            for x in seq:
                lx = x.lower()
                if lx in seen: continue
                seen.add(lx); out.append(x)
            return out
        return {"categorical": _dedupe(categorical), "numeric": _dedupe(numeric)}
    except Exception:
        return {"categorical": [], "numeric": []}
# ------------------ Financial News Pages ------------------

@app.get("/financial", response_class=HTMLResponse)
def financial_search(request: Request):
    """Financial news search page"""
    return templates.TemplateResponse("financial_index.html", {"request": request})

@app.get("/financial/visualizations", response_class=HTMLResponse)
def financial_visualizations(request: Request):
    """Financial news data visualizations page"""
    return templates.TemplateResponse("financial_visualizations.html", {"request": request})

# ------------------ Financial News API ------------------

@app.get("/financial/search")
def financial_search_api(
    query: str = Query(..., description="Text query for financial news search"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top records to return")
):
    results = search_financial_news(query, top_k)
    return {"results": results}

@app.get("/financial/api/chart-data")
def get_financial_chart_data(
    x_field: Optional[str] = Query(None, description="X-axis field (categorical): Category, Source, Date"),
    y_metric: str = Query("count", description="Y-axis metric: count | sum | avg"),
    value_field: Optional[str] = Query(None, description="Numeric field for sum/avg. Example: None for count"),
    limit: int = Query(10, description="Number of top items to show")
):
    if fin_df is None or fin_df.empty:
        return {"labels": [], "data": []}

    def col_for(df_in, name):
        lower = name.lower()
        for c in df_in.columns:
            if c.lower() == lower:
                return c
        return None

    if not x_field:
        return {"labels": [], "data": []}
    x_col = col_for(fin_df, x_field)
    if not x_col:
        return {"labels": [], "data": []}

    y_metric = (y_metric or "count").lower()
    if y_metric == "count":
        counts = fin_df[x_col].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    else:
        val_col = col_for(fin_df, value_field or "")
        if not val_col:
            return {"labels": [], "data": []}
        fin_df[val_col] = pd.to_numeric(fin_df[val_col], errors="coerce")
        grouped = fin_df.groupby(x_col)[val_col]
        agg = grouped.sum() if y_metric == "sum" else grouped.mean()
        agg = agg.sort_values(ascending=False).head(limit)
        return {"labels": agg.index.astype(str).tolist(), "data": agg.values.tolist()}

@app.get("/financial/api/fields")
def get_financial_fields():
    return _fields_for_df(fin_df if fin_df is not None else pd.DataFrame())
try:
    edu_data = joblib.load("app/models/education_models/education_policy_tfidf_matrix.pkl")
    edu_df = edu_data.get("df")
except Exception:
    edu_data = None
    edu_df = None

# CSV fallback for education dataset
if edu_df is None or getattr(edu_df, 'empty', True):
    try:
        edu_df = pd.read_csv("app/datasets/education_policies.csv")
    except Exception:
        edu_df = pd.DataFrame()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Landing page with policy domain selection"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthcare", response_class=HTMLResponse)
def healthcare_search(request: Request):
    """Healthcare policy search page"""
    return templates.TemplateResponse("healthcare_index.html", {"request": request})

@app.get("/visualizations", response_class=HTMLResponse)
def visualizations(request: Request):
    """Healthcare data visualizations page"""
    return templates.TemplateResponse("visualizations.html", {"request": request})

# ------------------ Healthcare API ------------------

@app.get("/search")
def search(
    query: str = Query(..., description="Text query for patient search"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top records to return"),
    condition: Optional[str] = Query(None, description="Filter: Medical Condition"),
    doctor: Optional[str] = Query(None, description="Filter: Doctor"),
    hospital: Optional[str] = Query(None, description="Filter: Hospital")
):
    results = search_records(query, top_k, condition, doctor, hospital)
    return {"results": results}

@app.get("/api/chart-data")
def get_chart_data(
    chart_type: Optional[str] = Query(None, description="Legacy: condition, doctor, hospital, age_group, gender, admission_type, test_results"),
    x_field: Optional[str] = Query(None, description="X-axis field (categorical). Example: Medical Condition, Doctor, Hospital, Age Group, Gender, Admission Type, Test Results"),
    y_metric: str = Query("count", description="Y-axis metric: count | sum | avg"),
    value_field: Optional[str] = Query(None, description="Numeric field for sum/avg. Example: Billing Amount, Age"),
    limit: int = Query(10, description="Number of top items to show")
):
    """API endpoint to get chart data with flexible axes and metrics."""

    # Helper to resolve a column name case-insensitively
    def col_for(df_in: pd.DataFrame, name: str) -> Optional[str]:
        if not name:
            return None
        lower = name.lower()
        for c in df_in.columns:
            if c.lower() == lower:
                return c
        return None

    # Back-compat mapping from chart_type to x_field
    if not x_field and chart_type:
        mapping = {
            "condition": "Medical Condition",
            "doctor": "Doctor",
            "hospital": "Hospital",
            "age_group": "Age Group",
            "gender": "Gender",
            "admission_type": "Admission Type",
            "test_results": "Test Results",
        }
        x_field = mapping.get(chart_type)

    if not x_field:
        return {"labels": [], "data": []}

    df_copy = df.copy()

    # Special handling for Age Group
    if x_field.lower() in {"age group", "age_group"}:
        df_copy["Age"] = pd.to_numeric(df_copy.get("Age"), errors="coerce")
        df_copy["Age Group"] = pd.cut(df_copy["Age"], bins=[0, 20, 40, 60, 80, 200], labels=["0-20", "21-40", "41-60", "61-80", "81+"])
        x_col = "Age Group"
    else:
        x_col = col_for(df_copy, x_field)

    if not x_col:
        return {"labels": [], "data": []}

    y_metric = (y_metric or "count").lower()

    if y_metric == "count":
        series = df_copy[x_col].value_counts().head(limit)
        labels = series.index.tolist()
        values = series.values.tolist()
    else:
        # Require a value_field for sum/avg
        val_col = col_for(df_copy, value_field or "")
        if not val_col:
            # default to Billing Amount if exists
            val_col = col_for(df_copy, "Billing Amount")
        if not val_col:
            return {"labels": [], "data": []}
        df_copy[val_col] = pd.to_numeric(df_copy[val_col], errors="coerce")
        grouped = df_copy.groupby(x_col)[val_col]
        agg = grouped.sum() if y_metric == "sum" else grouped.mean()
        # sort descending by metric, take top
        agg = agg.sort_values(ascending=False).head(limit)
        labels = agg.index.astype(str).tolist()
        values = agg.values.tolist()

    return {"labels": labels, "data": values}

@app.get("/api/billing-stats")
def get_billing_stats():
    """Get billing amount statistics for histogram"""
    df_copy = df.copy()
    df_copy["Billing Amount"] = pd.to_numeric(df_copy["Billing Amount"], errors="coerce")
    billing_data = df_copy["Billing Amount"].dropna().tolist()
    return {"data": billing_data}

@app.get("/api/fields")
def get_healthcare_fields():
    return _fields_for_df(df if df is not None else pd.DataFrame())

# ------------------ Education Pages ------------------

@app.get("/education", response_class=HTMLResponse)
def education_search(request: Request):
    """Education domain search page"""
    return templates.TemplateResponse("education_index.html", {"request": request})

@app.get("/education/visualizations", response_class=HTMLResponse)
def education_visualizations(request: Request):
    """Education data visualizations page"""
    return templates.TemplateResponse("education_visualizations.html", {"request": request})


# ------------------ Education Quantum Section ------------------

@app.get("/education/quantum", response_class=HTMLResponse)
def education_quantum(request: Request):
    """Education 'Quantum' TF-IDF exploration page"""
    return templates.TemplateResponse("education_quantum_index.html", {"request": request})


@app.get("/education/quantum/search")
def education_quantum_search_api(
    query: str = Query(..., description="Text query for education quantum search"),
    top_k: int = Query(3, ge=1, le=20, description="Number of top records to return")
):
    results = search_quantum(query, top_k)
    # Normalize keys and include optional metadata if present
    normalized = []
    for r in results:
        item = {
            "Index": r.get("index"),
            "Similarity": float(r.get("similarity", 0.0)),
        }
        # Pass through common optional fields when available
        for k in ("Title", "Sector", "Year", "Summary"):
            if k in r:
                item[k] = r[k]
        normalized.append(item)
    return {"results": normalized}

# ------------------ Healthcare Quantum ------------------

@app.get("/healthcare/quantum", response_class=HTMLResponse)
def healthcare_quantum(request: Request):
    return templates.TemplateResponse("healthcare_quantum_index.html", {"request": request})

@app.get("/healthcare/quantum/search")
def healthcare_quantum_search_api(
    query: str = Query(..., description="Text query for healthcare quantum search"),
    top_k: int = Query(3, ge=1, le=20, description="Number of top records to return")
):
    if not search_quantum_healthcare:
        return {"results": []}
    results = search_quantum_healthcare(query, top_k)
    return {"results": results}

# ------------------ Education API ------------------

@app.get("/education/search")
def education_search_api(
    query: str = Query(..., description="Text query for education search"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top records to return"),
    # Legacy filters (backward compatibility):
    program: Optional[str] = Query(None, description="Legacy filter: Program -> maps to Sector"),
    institution: Optional[str] = Query(None, description="Legacy filter: Institution -> maps to Region"),
    gender: Optional[str] = Query(None, description="Legacy filter: Gender (ignored)"),
    # New policy filters:
    sector: Optional[str] = Query(None, description="Filter: Sector"),
    region: Optional[str] = Query(None, description="Filter: Region"),
    status: Optional[str] = Query(None, description="Filter: Status"),
    year: Optional[str] = Query(None, description="Filter: Year")
):
    # Prefer new filters; fall back to legacy if new ones not provided
    sector = sector or program
    region = region or institution
    # gender is not used in the new policy dataset

    # Use the new policy-aware search; wrapper still exists for legacy callers
    results = search_policies(query=query, top_k=top_k, sector=sector, region=region, status=status, year=year)
    return {"results": results}

@app.get("/education/api/chart-data")
def get_education_chart_data(
    chart_type: Optional[str] = Query(None, description="Legacy: sector, region, status, year"),
    x_field: Optional[str] = Query(None, description="X-axis field (categorical): Sector, Region, Status, Year"),
    y_metric: str = Query("count", description="Y-axis metric: count | sum | avg"),
    value_field: Optional[str] = Query(None, description="Numeric field for sum/avg. Example: Funding (USD M)"),
    limit: int = Query(10, description="Number of top items to show")
):
    if edu_df is None or edu_df.empty:
        return {"labels": [], "data": []}

    # Helper to find column by lowercase name
    def col_for(df_in: pd.DataFrame, name: str):
        lower = name.lower()
        for c in df_in.columns:
            if c.lower() == lower:
                return c
        return None

    # Map legacy chart_type to x_field
    if not x_field and chart_type:
        x_field = chart_type

    if not x_field:
        return {"labels": [], "data": []}

    x_col = col_for(edu_df, x_field)
    if not x_col:
        return {"labels": [], "data": []}

    y_metric = (y_metric or "count").lower()

    if y_metric == "count":
        counts = edu_df[x_col].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    else:
        # sum/avg requires a numeric value_field
        val_col = col_for(edu_df, value_field or "")
        if not val_col:
            # default to funding column
            val_col = None
            for c in edu_df.columns:
                if c.lower() == "funding_million_usd":
                    val_col = c
                    break
        if not val_col:
            return {"labels": [], "data": []}
        df_copy = edu_df.copy()
        df_copy[val_col] = pd.to_numeric(df_copy[val_col], errors="coerce")
        grouped = df_copy.groupby(x_col)[val_col]
        agg = grouped.sum() if y_metric == "sum" else grouped.mean()
        agg = agg.sort_values(ascending=False).head(limit)
        return {"labels": agg.index.astype(str).tolist(), "data": agg.values.tolist()}

@app.get("/education/api/fees-stats")
def get_education_fees_stats():
    """Return distribution for funding (million USD) to drive D3 histogram."""
    if edu_df is None or edu_df.empty:
        return {"data": []}
    # Find funding column case-insensitively
    funding_col = None
    for c in edu_df.columns:
        if c.lower() == "funding_million_usd":
            funding_col = c
            break
    if not funding_col:
        return {"data": []}
    df_copy = edu_df.copy()
    df_copy[funding_col] = pd.to_numeric(df_copy[funding_col], errors="coerce")
    funding_data = df_copy[funding_col].dropna().tolist()
    return {"data": funding_data}

@app.get("/education/api/fields")
def get_education_fields():
    return _fields_for_df(edu_df if edu_df is not None else pd.DataFrame())

# ------------------ Error handlers ------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    status = getattr(exc, 'status_code', 500)
    template = "error.html"
    try:
        return templates.TemplateResponse(template, {"request": request, "status": status, "detail": str(exc.detail)})
    except Exception:
        # Fallback plain text
        return HTMLResponse(content=f"Error {status}: {exc.detail}", status_code=status)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        return templates.TemplateResponse("error.html", {"request": request, "status": 422, "detail": str(exc)})
    except Exception:
        return HTMLResponse(content=f"Validation Error: {exc}", status_code=422)
