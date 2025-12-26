# Fraud Detection MLOps System 🚨

An end-to-end, production-grade fraud detection system built with a strong focus on **decision optimization, explainability, and deployment**.
This project goes beyond model training to address real-world challenges such as class imbalance, threshold selection, training–serving skew, and model governance.

🔗 **Live Demo (Swagger UI):**
[https://fraud-detection-mlops-f98a.onrender.com/docs/](https://fraud-detection-mlops-f98a.onrender.com/docs/)

<img width="1577" height="812" alt="image" src="https://github.com/user-attachments/assets/6b75a66b-e3f7-44d8-be8c-f1d327ad3daa" />

---

## 🔍 Problem Statement

Credit card fraud detection is a **highly imbalanced classification problem** where:

* Fraud rates are extremely low
* False positives create customer friction
* Accuracy is a misleading metric

The goal is not just to build a classifier, but to design a **decision system** that:

* Maximizes fraud capture (recall)
* Controls false positives (FPR)
* Is explainable and deployable

---

## 🧠 Key Design Decisions

### 1️⃣ Metric Discipline

* Used **PR-AUC** instead of accuracy or ROC-AUC
* Explicitly analyzed metric behavior under extreme class imbalance

### 2️⃣ Cost-Aware Learning

* Simulated realistic fraud rates (1% / 5%)
* Used class-weighted models to reflect asymmetric costs

### 3️⃣ Decision Threshold Optimization

* Rejected the default 0.5 probability cutoff
* Selected an operational threshold (**0.006**) based on:

  * Maximum recall
  * False Positive Rate ≤ 0.1%

📌 *This threshold is treated as a first-class artifact, not a magic constant.*

### 4️⃣ Training–Serving Consistency

* Identified and fixed feature schema mismatch (`id` leakage)
* Ensured deterministic feature ordering and schema alignment
* Prevented silent inference failures common in production ML systems

---

## 🏗️ System Architecture

```
Raw Data
   │
   ▼
Data Preprocessing & Imbalance Simulation
   │
   ▼
Model Training (Logistic Regression, XGBoost)
   │
   ▼
MLflow Experiment Tracking
   │
   ▼
Threshold Optimization (Recall vs FPR)
   │
   ▼
FastAPI Inference Service
   │
   ├── /predict  → Probability-based fraud decision
   └── /explain  → SHAP-based feature attribution
   │
   ▼
Dockerized Deployment → Render (Cloud)
```

---

## 🤖 Models Used

| Model               | Purpose                              |
| ------------------- | ------------------------------------ |
| Logistic Regression | Interpretable baseline               |
| XGBoost             | High-performance fraud ranking model |

XGBoost was selected for deployment due to superior **PR-AUC** and ranking performance.

---

## 📊 Explainability with SHAP

To support governance and human review, the system exposes a dedicated `/explain` endpoint using **SHAP**.

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

This enables:

* Feature-level attribution per transaction
* Regulatory explainability
* Human-in-the-loop decision support

---

## 🚀 API Endpoints

| Endpoint   | Description                  |
| ---------- | ---------------------------- |
| `/health`  | Service health check         |
| `/predict` | Fraud probability + decision |
| `/explain` | SHAP-based explanation       |

Swagger UI:
👉 [https://fraud-detection-mlops-f98a.onrender.com/docs/](https://fraud-detection-mlops-f98a.onrender.com/docs/)

---

## 🐳 Deployment & MLOps

* FastAPI for inference
* Dockerized service
* Deployed on **Render**
* Cloudflare-proxied public endpoint
* Structured logging for monitoring and alert-rate tracking

<img width="1580" height="761" alt="image" src="https://github.com/user-attachments/assets/c1a2a1e5-fbce-4aa2-a620-1de87155c664" />

This demonstrates full **model-to-production lifecycle ownership**.

---

## 🧪 Testing & Validation

* Tested locally via Swagger UI
* Verified via `curl`
* Validated using known fraud and non-fraud samples
* Confirmed behavior under low-risk and high-risk inputs

---

## 📁 Project Structure

```
fraud-detection-mlops/
├── api/               # FastAPI application
├── src/               # Training, evaluation, utilities
├── artifacts/         # Trained model & threshold
├── docker/            # Dockerfile
├── notebooks/         # Analysis & threshold tuning
├── requirements.txt
└── README.md
```

---

## 🧾 Key Learnings

* Fraud detection is a **decision problem**, not just a classification task
* Threshold selection matters more than raw accuracy
* Training–serving skew is a real production risk
* Explainability is essential for trust and governance
* Deployment completes the ML lifecycle

---

## 🔮 Future Improvements

* Batch prediction endpoint
* Drift detection on input distributions
* CI pipeline for automated builds
* Cloud deployment on AWS ECS

---

## 👤 Author

Built by **[Shreyas Gaikwad]**
End-to-End Machine Learning & MLOps

