🧠 Policy Nav AI

Policy Nav AI is an intelligent policy search and recommendation system built using FastAPI, HTML, and CSS.
It helps users explore domain-specific policies in areas like Education, Healthcare, Finance, and Quantum Computing using pre-trained machine learning models.

This project demonstrates how AI-powered models can be integrated into a FastAPI backend with a simple, elegant web-based frontend interface.

🌐 Live Demo: https://policy-nav-ai-1.onrender.com/

⚙️ Features

✅ FastAPI Backend — Handles data processing and model predictions.
✅ Frontend (HTML + CSS) — Provides a responsive, user-friendly interface.
✅ Pre-trained Domain Models — Supports multiple AI models for different policy sectors.
✅ Search Functionality — Quickly find and rank relevant policy data.
✅ Lightweight Deployment — Easy to run locally or deploy online (e.g., Render, Vercel).

🛠️ Project Setup Guide

Follow these steps to set up and run the project successfully on your local machine 👇

🧩 Step 1: Create a Virtual Environment
python -m venv virtual

📂 Step 2: Navigate into the Virtual Environment Folder
cd virtual

🌐 Step 3: Clone the GitHub Repository
git clone https://github.com/<your-username>/policy-nav-ai.git


Replace <your-username> with your actual GitHub username.

⚡ Step 4: Activate the Virtual Environment

For Windows:

cd Scripts
activate


For macOS/Linux:

source bin/activate


You should now see (virtual) at the start of your terminal line.

📦 Step 5: Install Project Dependencies
pip install -r requirements.txt


This installs FastAPI, Uvicorn, Pandas, NumPy, Scikit-learn, and other dependencies.

🧭 Step 6: Navigate to the Backend Directory
cd ..
cd Policy_Search_App/backend

🚀 Step 7: Run the FastAPI Server
python -m uvicorn main:app --reload


If everything is correct, you’ll see:

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

🌍 Step 8: Open the Application

Visit:
👉 http://127.0.0.1:8000

You should see the Policy Nav AI interface running successfully.

🔚 Step 9: Deactivate the Virtual Environment
deactivate


This safely exits the virtual environment.

🧰 Tech Stack
Component	Technology
Backend	FastAPI
Frontend	HTML, CSS
Server	Uvicorn
Language	Python 3.9+
ML Models	scikit-learn, Qiskit, NumPy, Pandas
💡 Example Usage

Choose a domain — Education, Healthcare, Finance, or Quantum.

Enter a keyword or phrase related to a policy.

The AI model retrieves and ranks the most relevant policy information.

🧾 Commands Summary
Action	Command
Create virtual environment	python -m venv virtual
Enter virtual environment	cd virtual
Clone repository	git clone https://github.com/<your-username>/policy-nav-ai.git
Activate (Windows)	cd Scripts && activate
Install dependencies	pip install -r requirements.txt
Run backend server	python -m uvicorn main:app --reload
Deactivate environment	deactivate
🌐 Live Deployment (Render)

Your project is live and accessible at:
👉 https://policy-nav-ai-1.onrender.com/

Hosted using Render Web Services with FastAPI and HTML frontend support.

👨‍💻 Author

Pavan Kumar Bushigampala
🧩 Data Science & AI Enthusiast | Full Stack Developer | FastAPI + React Developer
📧 Email: pavankumarbushigampala@gmail.com

🌐 GitHub: https://github.com/<your-username>
