Project: PolicyNav (Policy Search App)

Goal
-----
Help a developer (or AI coding agent) make safe, small, high-value code changes in this FastAPI-based policy search app. Keep edits focused and low-risk: prefer tests, small refactors, documentation, and wiring fixes over large model/algorithm changes.

Quick architecture (why/how)
---------------------------
- FastAPI app entrypoint: `main.py` — mounts static/templates, loads model artifacts from `models/`, and provides routes for HTML pages and JSON visualization APIs.
- Preprocessing: `preprocessing.py` — canonical small helper that builds `text_for_nlp` used by the quantum path and templates.
- Models & artifacts: `models/` — contains pickled TF-IDF vectorizers, matrices, and an optional pickled quantum kernel. `MODEL_CONFIG` in `main.py` maps logical model keys (nlp1, nlp2, nlp3, qnlp) to their assets and CSV datasets.
- Datasets: `datasets/` — CSVs used to populate pages and visualization endpoints.
- Templates: `templates/` and partials under `templates/partials/` — used for rendering search pages and result fragments.

Key developer workflows
-----------------------
- Local run (Windows PowerShell):
  1) Create & activate venv: `python -m venv .venv` then `..\.venv\Scripts\Activate.ps1`
  2) Install deps: `pip install -r requirements.txt`
  3) Start dev server: `uvicorn main:app --reload --port 8000`

- Model artifacts: Changes to TF-IDF/vectorizer artifacts must update `MODEL_CONFIG` paths in `main.py` if names change. The server loads them at startup and will raise if none load.

- Debugging runtime errors: `main.py` prints full traceback to console and returns friendly error pages from `templates/partials/error.html`. Check console logs for detailed stack traces.

Project-specific conventions & patterns
--------------------------------------
- Model identification: The app uses logical keys ("nlp1", "nlp2", "nlp3", "qnlp") — use these when adding routes, templates, or tests.
- Flexible artifact formats: `main.py` accepts multiple saved matrix formats (numpy arrays, scipy sparse, pandas DataFrame, dict wrappers, list-of-dicts). When adding a new model artifact, prefer saving the raw sparse matrix or a dict with a `"matrix"` key so `_normalize_matrix` finds it.
- Quantum path: `qnlp` expects a pickled kernel-like object with an `.evaluate()` method or falls back to an internal `FidelityQuantumKernel`. The preprocessing pipeline must populate `text_for_nlp` (see `preprocessing.py`).
- Templates: Model pages use `TEMPLATE_MAP` in `main.py` to select template filenames (e.g., `nlp2` -> `education.html`). Add templates following the existing pattern and include `partials/results.html` to render search results.
- Defensive coding: The code prefers returning HTML error fragments to clients while logging full tracebacks. Maintain that pattern for consistency.

Examples & actionable edits
---------------------------
- Add a new model "nlp4":
  1) Add files: `models/nlp4_vectorizer.pkl`, `models/nlp4_tfidf_matrix.pkl`, and `datasets/nlp4.csv`.
  2) Update `MODEL_CONFIG` in `main.py` with a new key `"nlp4"` pointing at the files.
  3) Add an entry to `TEMPLATE_MAP` and a template `templates/<name>.html` if you want a model page.

- Fix a broken template render: Look at the server console for the traceback (main.py logs it). Template rendering errors are handled in `_make_model_page` — replicate its pattern when adding new dynamic pages.

Safe edit rules for AI agents (apply these strictly)
--------------------------------------------------
1) Do not change model-loading semantics in one PR. If you must, add backward-compatible handling and tests.
2) Prefer small, well-scoped edits (<200 LOC) with unit tests where appropriate. For template-only changes, a manual smoke-run is acceptable.
3) When editing `MODEL_CONFIG`, guard with clear error messaging; keep the existing verification/logging outputs.
4) When modifying NLP/vectorizer behavior, update `preprocessing.py` and ensure `text_for_nlp` remains present — many code paths rely on that column.
5) Add integration tests for search endpoints and visualization APIs rather than changing HTML rendering logic.

Files to inspect for context
----------------------------
- `main.py` — primary logic, routing, model loading, and the place to add `MODEL_CONFIG` entries.
- `preprocessing.py` — text cleaning and `text_for_nlp` contract.
- `templates/partials/results.html` and `templates/partials/error.html` — result and error rendering patterns.
- `models/` — example pickled artifacts showing expected shapes/formats.
- `datasets/*.csv` — sample data shaping expectations (columns like `title`, `full_text`, `stakeholders`).

If something is missing
----------------------
- If a model artifact is not present, prefer adding a fallback in `main.py` that builds a TF-IDF from the dataset (there's an existing pattern used for the quantum fallback). Mirror that pattern rather than replacing it.
- If you need to change how similarity scores are computed, keep numeric stability checks (e.g., clamping top_k, handling sparse vs dense) and log tracebacks.

Contact / iteration
-------------------
If any section is unclear or you need more examples (tests, specific template snippets, or a sample new model addition), tell me which part and I'll update this file.
