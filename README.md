# ThreatVision AI
AI-Driven Intrusion Detection and Cyber Threat Analysis System

ThreatVision AI is a machine learning–based intrusion detection system designed to analyze network traffic and detect malicious activities.
The project focuses on applying AI techniques to cybersecurity using real-world datasets (CICIDS 2017 & 2018) and provides a complete workflow from data preprocessing and model training to real-time inference through an API.

It is built as a research-oriented and practical implementation of AI-powered threat detection.

---

## 🔬 Objectives

- Detect malicious network traffic using machine learning models
- Study behavior-based intrusion detection
- Build a scalable AI pipeline for cybersecurity applications
- Provide a real-time prediction interface through an API

---

## ⚙️ Core Components

- Data preprocessing and feature engineering
- Model training and evaluation
- Model serialization and loading
- Real-time prediction API
- Logging and configuration management

---

## 🛠️ Technology Stack

- Python – Main programming language
- Scikit-learn – Machine learning algorithms
- Pandas, NumPy – Data processing
- FastAPI / Flask – API layer
- Pickle (.pkl) – Model persistence
- Jupyter Notebook – Experimentation & research
- YAML – Configuration and logging setup

---

## 📂 Project Structure

```text
ThreatVision_AI/
│   README.md
│   requirements.txt
│
├── api/                 # API for real-time predictions
│   ├── threatvision_api.py
│   ├── routes/
│   └── utils/
│
├── config/              # Configuration and logging files
│
├── data/                # CICIDS datasets
│
├── logs/                # Runtime logs
│
├── models/              # Trained ML models (.pkl)
│
├── notebooks/           # Research and experimentation notebooks
│
├── outputs/             # Evaluation results, charts, reports
│
└── scripts/             # Data processing and model training scripts
```

---

## 🧠 System Workflow

1. Load raw network traffic data from CICIDS datasets
2. Perform data cleaning and preprocessing
3. Apply feature engineering
4. Train ML models on labeled traffic data
5. Evaluate and select the best-performing model
6. Save the trained model as a `.pkl` file
7. Load the model inside the API service
8. Perform real-time predictions on incoming data

---

## ▶️ Installation

```bash
git clone https://github.com/songarapankaj1304-cmd/ThreatVision-AI.git
cd ThreatVision-AI
pip install -r requirements.txt
```

---

## 🚀 Run the API

```bash
# one-time: create sample artifacts if models/*.pkl are missing
python models/create_sample_models.py

python api/threatvision_api.py
```

The API will start and allow real-time prediction of malicious or benign traffic.

---

## 🧪 Train the Model

```bash
python scripts/train_model.py
```

This script trains the ML model and saves the final version inside the `models/` directory.

---

## 📊 Datasets

The project is based on:

- CICIDS 2017
- CICIDS 2018

These datasets contain realistic attack scenarios such as:
- DDoS
- Port scanning
- Web attacks
- Infiltration
- Brute-force attacks

They are widely used in intrusion detection research.

---

## 🎯 Research & Practical Use

- AI-based Intrusion Detection Systems (IDS)
- SOC training environments
- Cybersecurity research projects
- Academic and final-year projects
- Threat behavior analysis

---

## ⚠️ Disclaimer

This project is intended strictly for educational and research purposes.
It must only be used in authorized environments.
The author is not responsible for any misuse.

---

## 👤 Author

Pankaj Songara
Cybersecurity Student | AI & Machine Learning in Security
