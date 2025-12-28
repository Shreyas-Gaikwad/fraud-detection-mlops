# Fraud Detection MLOps System

An end-to-end, production-grade fraud detection system built with a strong focus on **decision optimization, explainability, and deployment**.
This project goes beyond model training to address real-world challenges such as class imbalance, threshold selection, training–serving skew, and model governance.

Live Demo (Swagger UI):
[https://fraud-detection-mlops-f98a.onrender.com/docs/](https://fraud-detection-mlops-f98a.onrender.com/docs/)

**Tech Stack:**  
Python · Pandas · NumPy · Scikit-learn · XGBoost · MLflow · SHAP · FastAPI · Docker · Render

<img width="1577" height="812" alt="image" src="https://github.com/user-attachments/assets/6b75a66b-e3f7-44d8-be8c-f1d327ad3daa" />

---

## Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where:

* Fraud rates are extremely low
* False positives create customer friction
* Accuracy is a misleading metric

The goal is not just to build a classifier, but to design a **decision system** that:

* Maximizes fraud capture (recall)
* Controls false positives (FPR)
* Is explainable and deployable

---

## Features

* End-to-end fraud detection pipeline (training to deployment)
* Cost-sensitive learning for extreme class imbalance
* Decision-threshold optimization based on recall–FPR trade-offs
* MLflow experiment tracking
* FastAPI-based inference service
* SHAP-based local explainability via `/explain` endpoint
* Dockerized application
* Public cloud deployment on Render

---

## System Architecture

```
Raw Data
   |
   v
Data Preprocessing & Imbalance Simulation
   |
   v
Model Training (Logistic Regression, XGBoost)
   |
   v
MLflow Experiment Tracking
   |
   v
Threshold Optimization (Recall vs FPR)
   |
   v
FastAPI Inference Service
   |
   |-- /predict  -> Probability-based fraud decision
   |-- /explain  -> SHAP-based feature attribution
   |
   v
Dockerized Deployment -> Render (Cloud)
```

---

## Models Used

| Model               | Purpose                              |
| ------------------- | ------------------------------------ |
| Logistic Regression | Interpretable baseline               |
| XGBoost             | High-performance fraud ranking model |

XGBoost was selected for deployment due to superior PR-AUC and probability ranking performance under class imbalance.

---

## Explainability with SHAP

To support governance and human review, the system exposes a dedicated `/explain` endpoint using SHAP.

Example response:

```json
{
  "fraud_probability": 0.999284,
  "decision": true,
  "top_contributing_features": {
    "V14": 3.381843,
    "V8": -1.714934,
    "V4": 1.412652
  }
}
```

This provides feature-level attribution for each prediction and enables transparent, auditable fraud decisions.

---

## API Endpoints

| Endpoint   | Description                    |
| ---------- | ------------------------------ |
| `/health`  | Service health check           |
| `/predict` | Fraud probability and decision |
| `/explain` | SHAP-based explanation         |

Swagger UI:
[https://fraud-detection-mlops-f98a.onrender.com/docs/](https://fraud-detection-mlops-f98a.onrender.com/docs/)

---

## Deployment & MLOps

* FastAPI for inference
* Dockerized service
* Deployed on **Render**
* Cloudflare-proxied public endpoint
* Structured logging for monitoring and alert-rate tracking

<img width="1580" height="761" alt="image" src="https://github.com/user-attachments/assets/c1a2a1e5-fbce-4aa2-a620-1de87155c664" />

This demonstrates full **model-to-production lifecycle ownership**.

---

## Data

### Dataset Source

The dataset used in this project is the **[Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)**, publicly available on Kaggle.

Due to licensing and size constraints, the dataset is not included in this repository. Users are expected to download the dataset separately and place it in the appropriate data directory before training.

### Dataset Description

* Contains anonymized transaction features (`V1`–`V28`) derived via PCA
* Includes transaction `Amount`
* Highly imbalanced target variable (`Class`)
* Identifier column (`id`) is removed during training to prevent leakage

The dataset is used to simulate realistic fraud rates and evaluate model behavior under operational constraints.

---

## Project Structure

```
fraud-detection-mlops/
├── api/               # FastAPI application
├── src/               # Training, evaluation, utilities
├── artifacts/         # Trained model and decision threshold
├── docker/            # Dockerfile
├── notebooks/         # Analysis and threshold tuning
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shreyas-Gaikwad/fraud-detection-mlops.git
cd fraud-detection-mlops
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## Usage

### Run locally

```bash
uvicorn api.app:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

### Run with Docker

```bash
docker build -t fraud-detection-mlops -f docker/Dockerfile .
docker run -p 8000:8000 fraud-detection-mlops
```

---

## Contributing

Contributions are welcome.

Please submit pull requests or open issues for:

* Bug reports
* Feature requests
* Documentation improvements

---

## Author

Built by **[Shreyas Gaikwad]**

Focus: End-to-End Machine Learning and MLOps
