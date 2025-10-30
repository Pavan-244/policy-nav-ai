# PolicyNav - AI-Powered Policy Search Application

## Live Demo
🚀 [View Live Application](https://your-app-name.onrender.com)

## Overview
PolicyNav is an AI-powered public policy navigation tool that uses TF-IDF and quantum-inspired NLP to search across multiple policy domains including healthcare, education, finance, and quantum education policies.

## Features
- 🔍 **Multi-Domain Search**: Search across health, education, financial, and quantum policy datasets
- 📊 **Interactive Visualizations**: Explore data with charts, word clouds, and relationship graphs
- 🎯 **Smart Similarity Matching**: Advanced TF-IDF and quantum kernel-based similarity scoring
- 💡 **Professional UI**: Clean, responsive interface with card-based results
- ⚡ **Fast Performance**: Optimized with sampling and efficient matrix operations

## Tech Stack
- **Backend**: FastAPI, Python 3.10
- **Frontend**: HTMX, Jinja2 Templates
- **ML/NLP**: scikit-learn, TF-IDF, Custom Quantum Kernels
- **Data Processing**: pandas, numpy, scipy
- **Visualizations**: Chart.js, D3.js

## Local Development

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Pavan-244/policy-nav-ai.git
cd policy-nav-ai/Policy_Search_App
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn backend.main:app --reload --port 8000
```

5. Open your browser:
```
http://localhost:8000
```

## Deployment on Render

### Quick Deploy
1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `policy-nav-ai`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"

### Environment Variables (Optional)
If needed, you can set these in Render:
- `PYTHON_VERSION`: `3.10.0`
- `PORT`: Auto-set by Render

## Project Structure
```
Policy_Search_App/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models/              # TF-IDF model artifacts
│   └── datasets/            # Policy CSV datasets
├── frontend/
│   ├── static/
│   │   ├── style.css        # Application styles
│   │   └── favicon.svg      # App icon
│   └── templates/           # Jinja2 HTML templates
├── requirements.txt         # Python dependencies
├── Procfile                # Render deployment config
└── runtime.txt             # Python version
```

## Models

### NLP Models (nlp1, nlp2, nlp3)
- Health Policies (nlp1): 55k+ patient records, optimized sampling
- Education Policies (nlp2): 500+ education reforms
- Financial News (nlp3): 3k+ market events

### Quantum Model (qnlp)
- Quantum-inspired kernel using cosine similarity
- Normalized feature vectors with π-scaling
- Education policy dataset

## API Endpoints

### Search
- `POST /search/{model_key}` - Search policies
  - Parameters: `query`, `top_k`

### Visualization
- `GET /api/visualize/{model_key}/columns` - Get dataset columns
- `GET /api/visualize/{model_key}/summary` - Get column summary
- `GET /api/visualize/{model_key}/nlp/terms` - Get top terms
- `GET /api/visualize/{model_key}/nlp/cooc` - Get word co-occurrence

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Author
**Pavan Kumar**
- GitHub: [@Pavan-244](https://github.com/Pavan-244)

## Acknowledgments
- FastAPI framework for the excellent web framework
- scikit-learn for ML utilities
- HTMX for seamless frontend interactions
