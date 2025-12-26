FRAUD DETECTION MLOPS SYSTEM

An end-to-end, production-grade fraud detection system with a strong focus on
decision optimization, explainability, and deployment. This project goes beyond
model training to address real-world challenges such as class imbalance,
threshold selection, training–serving skew, and model governance.

Live Demo (Swagger UI):
https://fraud-detection-mlops-f98a.onrender.com/docs/


==================================================
PROBLEM STATEMENT
==================================================

Credit card fraud detection is a highly imbalanced classification problem where:
- Fraud cases are extremely rare
- False positives create customer friction
- Accuracy is a misleading metric

The goal of this project is not just to build a classifier, but to design a
decision system that:
- Maximizes fraud capture (recall)
- Controls false positives (FPR)
- Is explainable and deployable in production


==================================================
KEY DESIGN DECISIONS
==================================================

1) Metric Discipline
-------------------
- Used Precision-Recall AUC (PR-AUC) instead of accuracy
- Explicitly analyzed metric behavior under extreme class imbalance

2) Cost-Aware Learning
---------------------
- Simulated realistic fraud rates (1% and 5%)
- Used class-weighted models to reflect asymmetric costs

3) Decision Threshold Optimization
----------------------------------
- Rejected the default 0.5 probability cutoff
- Selected an operational threshold of 0.006 based on:
  - Maximum recall
  - False Positive Rate ≤ 0.1%

The threshold is treated as a first-class artifact, not a magic constant.

4) Training–Serving Consistency
-------------------------------
- Identified and fixed feature schema mismatch (ID leakage)
- Enforced deterministic feature ordering
- Ensured strict alignment between training and inference pipelines


==================================================
SYSTEM ARCHITECTURE
==================================================

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
  |        |
  |        +-- /predict  (fraud probability + decision)
  |        +-- /explain  (SHAP-based explanations)
  |
  v
Dockerized Deployment -> Render (Cloud)


==================================================
MODELS USED
==================================================

- Logistic Regression
  Purpose: Interpretable baseline model

- XGBoost
  Purpose: High-performance fraud ranking model

XGBoost was selected for deployment due to superior PR-AUC and probability
ranking performance.


==================================================
EXPLAINABILITY WITH SHAP
==================================================

The system exposes a dedicated /explain endpoint using SHAP to provide
feature-level attribution for individual predictions.

Example response:

{
  "fraud_probability": 0.999284,
  "decision": true,
  "top_contributing_features": {
    "V14": 3.381843,
    "V8": -1.714934,
    "V4": 1.412652
  }
}

This enables:
- Regulatory explainability
- Human-in-the-loop review
- Model debugging and governance


==================================================
API ENDPOINTS
==================================================

/health
- Service health check

/predict
- Returns fraud probability and threshold-based decision

/explain
- Returns SHAP-based explanation for a single prediction

Swagger UI:
https://fraud-detection-mlops-f98a.onrender.com/docs/


==================================================
DEPLOYMENT & MLOPS
==================================================

- FastAPI for inference
- Dockerized service
- Deployed on Render
- Public cloud endpoint behind Cloudflare
- Structured logging for monitoring and alert-rate tracking

This demonstrates full ownership of the model-to-production lifecycle.


==================================================
TESTING & VALIDATION
==================================================

- Tested locally using Swagger UI
- Verified using curl requests
- Validated with known fraud and non-fraud samples
- Confirmed stable behavior under low-risk and high-risk inputs


==================================================
PROJECT STRUCTURE
==================================================

fraud-detection-mlops/
|
|-- api/            FastAPI application
|-- src/            Training, evaluation, utilities
|-- artifacts/      Trained model and decision threshold
|-- docker/         Dockerfile
|-- notebooks/      Analysis and threshold tuning
|-- requirements.txt
|-- README.txt


==================================================
KEY LEARNINGS
==================================================

- Fraud detection is a decision problem, not just classification
- Threshold selection matters more than raw accuracy
- Training–serving skew is a real production risk
- Explainability is critical for trust and governance
- Deployment completes the ML lifecycle


==================================================
FUTURE IMPROVEMENTS
==================================================

- Batch prediction endpoint
- Drift detection on input distributions
- CI pipeline for automated builds
- Deployment on AWS ECS


==================================================
AUTHOR
==================================================

Built by Shreyas Gaikwad
MCA (AI & ML)
End-to-End Machine Learning and MLOps
