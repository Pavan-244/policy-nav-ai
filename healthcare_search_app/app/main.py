from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from app.search_utils import search_records
import pandas as pd
import joblib

app = FastAPI(
    title="Healthcare Record Search API",
    description="Patient search API using TF-IDF & Cosine Similarity",
    version="1.0"
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load dataset for visualizations
data = joblib.load("app/models/healthcare_tfidf_matrix.pkl")
df = data["df"]

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/visualizations", response_class=HTMLResponse)
def visualizations(request: Request):
    return templates.TemplateResponse("visualizations.html", {"request": request})

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
    chart_type: str = Query(..., description="Type of chart: condition, doctor, hospital, age_group, gender"),
    limit: int = Query(10, description="Number of top items to show")
):
    """API endpoint to get chart data based on user selection"""
    
    if chart_type == "condition":
        counts = df["Medical Condition"].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    elif chart_type == "doctor":
        counts = df["Doctor"].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    elif chart_type == "hospital":
        counts = df["Hospital"].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    elif chart_type == "age_group":
        df_copy = df.copy()
        df_copy["Age"] = pd.to_numeric(df_copy["Age"], errors="coerce")
        df_copy["Age Group"] = pd.cut(df_copy["Age"], bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81+"])
        counts = df_copy["Age Group"].value_counts()
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    elif chart_type == "gender":
        counts = df["Gender"].value_counts()
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    elif chart_type == "admission_type":
        counts = df["Admission Type"].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    elif chart_type == "test_results":
        counts = df["Test Results"].value_counts().head(limit)
        return {"labels": counts.index.tolist(), "data": counts.values.tolist()}
    
    else:
        return {"labels": [], "data": []}

@app.get("/api/billing-stats")
def get_billing_stats():
    """Get billing amount statistics for histogram"""
    df_copy = df.copy()
    df_copy["Billing Amount"] = pd.to_numeric(df_copy["Billing Amount"], errors="coerce")
    billing_data = df_copy["Billing Amount"].dropna().tolist()
    return {"data": billing_data}
