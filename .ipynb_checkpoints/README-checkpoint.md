Healthcare Semantic Search & Visualization App
Overview
This project is a FastAPI-powered web application for semantic patient record search and interactive healthcare data visualizations. Users can search patient records using natural language and filter by fields like condition, doctor, or hospital. The app also provides powerful, user-controlled charts via Chart.js and D3.js, enabling insights into the dataset. The backend uses pre-trained TF-IDF models (via scikit-learn and joblib) for fast, meaningful retrieval. The UI is clean, responsive, and easy to use.

Features
Search patient records by semantic query, medical condition, doctor, and hospital

Dropdown menus for robust, error-free filter selection

Top-N result selection (choose number of results)

Interactive data visualizations:

Patient count by condition, doctor, hospital, age group, etc. (Chart.js)

Financial/billing histogram (D3.js)

Chart type and dataset field selectable by user

All major data fields shown in results: Name, Age, Gender, Condition, Dates, Medication, etc.

Modern responsive web design

API endpoints for custom integrations

Folder Structure

healthcare_search_app/
│
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application: routes, API, templates
│   ├── search_utils.py        # TF-IDF search logic
│   ├── models/                # ML models and data
│   │   ├── healthcare_vectorizer.pkl
│   │   └── healthcare_tfidf_matrix.pkl
│   ├── templates/             # HTML interface
│   │   ├── index.html
│   │   └── visualizations.html
│   ├── static/                # Static assets
│   │   ├── style.css
│   │   ├── chart.min.js       # Chart.js library
│   │   └── d3.min.js          # D3.js library
│   ├── requirements.txt       # All Python dependencies
│
└── README.md                  # Project documentation


Setup Instructions
1. Clone or Download Project
text
git clone <your-repository-url>
cd healthcare_search_app
2. Add Model Files
Put healthcare_vectorizer.pkl and healthcare_tfidf_matrix.pkl (generated from your Python data pipeline) into the app/models/ folder.

3. Install Dependencies
You need Python 3.8+

text
pip install -r app/requirements.txt
4. Run the Application
From your project root (healthcare_search_app/):

text
uvicorn app.main:app --reload
5. Visit Web Interface
Search: http://localhost:8000/

Visualizations: http://localhost:8000/visualizations

Usage
Semantic Search
Enter a query like "diabetes treatment with insulin" in the form

Select Condition, Doctor, Hospital from dropdowns (or leave blank for broad search)

Choose top N results

Review patient details, matching score, complete medical info

Visualizations
Select chart type and data field (e.g. Medical Conditions, Doctor, Hospital, Age Group, etc.)

Choose bar, pie, doughnut, or line chart

Instant chart updates from real data

Click "Load Billing Histogram" for interactive billing amount distribution (D3.js)

API Endpoint Reference
Endpoint	Description
/search	Patient records search (JSON output)
/visualizations	Interactive chart dashboard (HTML, JS)
/api/chart-data	REST API for chart data (used by Chart.js)
/api/billing-stats	REST API for histogram data (used by D3.js)

Customization Guide
Add dropdown options: Edit <select>s in index.html to match your dataset values

More visualizations: Extend /api/chart-data in main.py and update visualizations.html

Styling: Modify style.css for colors, spacing, typography

Deployment
For production, use:

text
uvicorn app.main:app --host 0.0.0.0 --port 80
Use nginx, Docker, or Gunicorn behind a proxy for advanced deployment

Acknowledgements
FastAPI

scikit-learn

Chart.js

D3.js

