🧠 Policy Nav AI

Policy Nav AI is an intelligent policy search and recommendation system built using FastAPI, HTML, and CSS.
It helps users explore domain-specific policies in areas like Education, Healthcare, Finance, and Quantum Computing using pre-trained models.

This project demonstrates how machine learning models can be integrated into a FastAPI backend with a simple web-based frontend interface.

⚙️ Features

🌐 FastAPI Backend — handles data processing and model responses.

🖥️ Frontend (HTML + CSS) — provides a clean, user-friendly interface.

🧩 Pre-trained Domain Models — supports multiple AI models for different policy sectors.

🗂️ Search Functionality — quickly find relevant policy data.

🚀 Lightweight Deployment — easy to run locally or host online.

🛠️ Project Setup Guide

Follow these steps exactly in order to set up and run the project successfully:

🧩 Step 1: Create a Virtual Environment

Create a new Python virtual environment in your project directory:

python -m venv virtual

📂 Step 2: Navigate into the Virtual Environment Folder

Move into the newly created virtual environment:

cd virtual

🌐 Step 3: Clone the GitHub Repository

Clone this repository inside the virtual environment folder:

git clone https://github.com/<your-username>/policy-nav-ai.git


Replace <your-username> with your actual GitHub username.

⚡ Step 4: Activate the Virtual Environment
🪟 On Windows:
cd Scripts
activate

🐧 On macOS/Linux:
source bin/activate


Once activated, your terminal will show (virtual) at the beginning of the line.

📦 Step 5: Install Project Dependencies

After activation, install all required libraries using:

pip install -r requirements.txt


This command installs FastAPI, Uvicorn, Pandas, NumPy, Scikit-learn, and other dependencies listed in requirements.txt.

🧭 Step 6: Navigate to the Backend Directory

After installing the dependencies, move to the backend folder:

cd ..
cd Policy_Search_App/backend

🚀 Step 7: Run the FastAPI Server

Start the development server using:

python -m uvicorn main:app --reload


If everything is set up correctly, you’ll see:

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

🌍 Step 8: Open the Application

Now open your web browser and go to:
👉 http://127.0.0.1:8000

You should see the Policy Nav AI interface running successfully.

🔚 Step 9: Deactivate the Virtual Environment

Once you finish testing or development, deactivate the environment using:

deactivate


This safely exits the virtual environment and returns your terminal to the normal system state.

📁 Project Structure
policy-nav-ai/
│
├── Policy_Search_App/
│   ├── backend/
│   │   ├── main.py
│   │   ├── models/
│   │   ├── templates/
│   │   └── static/
│   │
│   ├── frontend/
│   │   ├── templates/
│   │   ├── static/
│   │   └── datasets/
│
├── models/
│   ├── education_models/
│   ├── financial_models/
│   ├── healthcare_models/
│   └── quantum_models/
│
├── datasets/
├── requirements.txt
├── README.md
└── .gitignore

🧰 Tech Stack
Component	Technology
Backend	FastAPI
Frontend	HTML, CSS
Server	Uvicorn
Language	Python 3.9+
ML Models	scikit-learn, Qiskit, NumPy, Pandas
💡 Example Usage

Choose a domain (Education, Healthcare, Finance, or Quantum).

Enter a keyword or phrase related to a policy.

The AI model retrieves and ranks the most relevant policy information.

🧾 Commands Summary
Action	Command
Create virtual environment	python -m venv virtual
Enter virtual environment	cd virtual
Clone repository	git clone https://github.com/<your-username>/policy-nav-ai.git
Activate environment (Windows)	cd Scripts && activate
Install dependencies	pip install -r requirements.txt
Run backend server	python -m uvicorn main:app --reload
Deactivate environment	deactivate
👨‍💻 Author

Pavan Kumar Bushigampala
🧩 Data Science & AI Enthusiast | Full Stack Developer | FastAPI + React Developer