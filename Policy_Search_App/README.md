PolicyNav - Policy Search App

Overview

PolicyNav is a small FastAPI web application that provides semantic search over multiple policy datasets (education, healthcare, financial news) using TF-IDF models and an experimental quantum kernel-based similarity path.

Run locally

1. Create a virtual environment (recommended) and install dependencies:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2. Start the app with Uvicorn:

   uvicorn main:app --reload --port 8000

3. Open http://localhost:8000 in your browser.

Notes

- Model artifacts (pickled vectorizers/matrices and quantum kernel) are expected to be in the `models/` folder and are included in the repository.
- Datasets are under `datasets/` and are used to populate result pages and visualization endpoints.
- The quantum kernel path requires `qiskit-machine-learning` and access to a quantum simulator or the provided pickled kernel.
