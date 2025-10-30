# 🧠 **Policy Nav AI**

### 🔍 *Intelligent Policy Search and Recommendation System using FastAPI + AI Models*

---

## 🌐 **Live Project**

**🚀 Deployed URL:**
👉 [https://policy-nav-ai-1.onrender.com/](https://policy-nav-ai-1.onrender.com/)

**🖥️ Tech Stack:**

> FastAPI | HTML | CSS | Python | Scikit-learn | Render Cloud Hosting

---

## 🧩 **Project Overview**

**Policy Nav AI** is a smart, domain-based **policy recommendation system** that leverages pre-trained machine learning models to assist users in exploring and understanding policy-related data in multiple domains such as:

* 🏫 **Education**
* 🏥 **Healthcare**
* 💰 **Finance**
* ⚛️ **Quantum Computing**

The system provides **search-based insights** and **relevant policy retrieval** using vectorized document representations and domain-specific models.

---

## ✨ **Key Features**

| Feature                          | Description                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| 🌐 **FastAPI Backend**           | Efficient and lightweight Python backend to handle data and AI model responses.              |
| 🖥️ **Frontend (HTML + CSS)**    | Clean and responsive UI designed for simplicity and speed.                                   |
| 🧠 **Pre-trained Domain Models** | Supports multiple AI models trained for Education, Healthcare, Finance, and Quantum domains. |
| 🔍 **Policy Search**             | Search and retrieve relevant policy information in seconds.                                  |
| ⚙️ **Lightweight Deployment**    | Easily deployable on platforms like Render or run locally.                                   |
| 🧰 **Scalable Structure**        | Modular backend with clear folder separation for future scalability.                         |

---

## 🏗️ **Project Folder Structure**

```
Policy_Nav_AI/
│
├── backend/
│   ├── main.py                # FastAPI main application file
│   ├── models/                # Contains vector/matrix model files for each domain
│   ├── templates/             # HTML templates for frontend rendering
│   ├── static/                # CSS and other frontend assets
│   ├── requirements.txt       # List of required Python libraries
│
├── frontend/
│   ├── index.html             # Main landing page
│   ├── styles.css             # Core CSS file for styling
│
└── README.md                  # Documentation file
```

---

## ⚙️ **Project Setup Guide**

Follow these **step-by-step instructions** to run Policy Nav AI on your local system 👇

---

### 🧩 **Step 1: Create a Virtual Environment**

Create a Python virtual environment:

```bash
python -m venv virtual
```

---

### 📂 **Step 2: Navigate into the Virtual Environment Folder**

```bash
cd virtual
```

---

### 🌐 **Step 3: Clone the GitHub Repository**

```bash
git clone https://github.com/<your-username>/policy-nav-ai.git
```

> Replace `<your-username>` with your actual GitHub username.

---

### ⚡ **Step 4: Activate the Virtual Environment**

**On Windows:**

```bash
cd Scripts
activate
```

**On macOS/Linux:**

```bash
source bin/activate
```

✅ You’ll know it’s active when your terminal shows `(virtual)` at the beginning.

---

### 📦 **Step 5: Install Project Dependencies**

```bash
pip install -r requirements.txt
```

This installs:

* FastAPI
* Uvicorn
* Pandas
* NumPy
* Scikit-learn
* Jinja2

---

### 🧭 **Step 6: Navigate to the Backend Folder**

```bash
cd ..
cd Policy_Search_App/backend
```

---

### 🚀 **Step 7: Run the FastAPI Server**

```bash
python -m uvicorn main:app --reload
```

If successful, you’ll see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### 🌍 **Step 8: Open the Application in Browser**

Now open your browser and visit:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

You should now see the **Policy Nav AI** interface live on your local machine 🎉

---

### 🔚 **Step 9: Deactivate the Virtual Environment**

When finished:

```bash
deactivate
```

---

## 🌐 **Deployment on Render**

Your project is already **deployed on Render Cloud** — a free hosting platform for FastAPI apps.

🔗 **Live Link:** [https://policy-nav-ai-1.onrender.com/](https://policy-nav-ai-1.onrender.com/)

### ⚙️ Render Deployment Summary

| Step | Description                                                                      |
| ---- | -------------------------------------------------------------------------------- |
| 1️⃣  | Create a new **Web Service** on Render                                           |
| 2️⃣  | Connect your GitHub repository                                                   |
| 3️⃣  | Set environment as `Python 3`                                                    |
| 4️⃣  | In “Start Command”, use → `uvicorn backend.main:app --host 0.0.0.0 --port 10000` |
| 5️⃣  | Deploy — Render automatically detects and builds your FastAPI project            |

---

## 💡 **How It Works**

1. **User Input** — The user selects a domain and enters a keyword (e.g., *“Higher Education Funding”*).
2. **Backend Processing** — FastAPI routes the request to the corresponding domain model.
3. **Model Inference** — The vectorized model finds the most relevant policy entries.
4. **Results Displayed** — Results are rendered beautifully on the web interface.

---

## 🧾 **Common Commands Summary**

| Action                     | Command                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| Create virtual environment | `python -m venv virtual`                                         |
| Enter virtual environment  | `cd virtual`                                                     |
| Clone repository           | `git clone https://github.com/<your-username>/policy-nav-ai.git` |
| Activate (Windows)         | `cd Scripts && activate`                                         |
| Install dependencies       | `pip install -r requirements.txt`                                |
| Run server                 | `python -m uvicorn main:app --reload`                            |
| Deactivate                 | `deactivate`                                                     |

---

## 🧠 **Future Enhancements**

* 🌍 Add **database integration (MongoDB or PostgreSQL)**
* 🤖 Enable **LLM-powered search (e.g., GPT-based semantic search)**
* 🧾 Add **policy upload and retraining module**
* 📊 Build **interactive analytics dashboard**

---

## 🤝 **Contributing**

We welcome contributions!

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Submit a pull request

---

## 📸 **Preview (Example UI)**

> *(Add screenshots of your running app here — example placeholders)*

| Home Page                | Education Result Page     | Quantum Result Page      | Finanice Result Page      | Visulization Result Page  | 
| -------------------------| --------------------------| -------------------------| --------------------------| --------------------------|
| ![Home](assets/home.png) | ![Home](assets/home.png)  | ![Home](assets/home.png) | ![Home](assets/home.png)  | ![Home](assets/home.png)  |
---

## 👨‍💻 **Author**

**Pavan Kumar Bushigampala**
🧩 *Data Science & AI Enthusiast | Full Stack Developer | FastAPI + React Developer*
📧 **Email:** [pavankumarbushigampala@gmail.com](mailto:pavankumarbushigampala@gmail.com)
💻 **GitHub:** [https://github.com/<your-username>](https://github.com/Pavan-244)
🌐 **Project:** [https://policy-nav-ai-1.onrender.com/](https://policy-nav-ai-1.onrender.com/)

---

## 📜 **License**

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---
