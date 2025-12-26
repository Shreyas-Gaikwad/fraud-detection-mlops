import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.data import load_data
from src.utils import simulate_imbalance


TARGET = "Class"
AMOUNT = "Amount"

def temporal_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:]
    )

def train_logistic(fraud_rate=0.01):

    df = load_data("data/creditcard2023.csv")
    df = simulate_imbalance(df, TARGET, fraud_rate)

    train_df, val_df, test_df = temporal_split(df)

    DROP_COLS = [TARGET, "id"]

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=DROP_COLS)
    y_val = val_df[TARGET]

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ))
    ])

    with mlflow.start_run(run_name=f"logreg_fraudrate_{fraud_rate}"):

        pipeline.fit(X_train, y_train)

        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        metrics = {
            "precision": precision_score(y_val, val_preds),
            "recall": recall_score(y_val, val_preds),
            "f1": f1_score(y_val, val_preds),
            "roc_auc": roc_auc_score(y_val, val_probs),
            "pr_auc": average_precision_score(y_val, val_probs)
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "fraud_rate": fraud_rate,
            "class_weight": "balanced"
        })

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        print(metrics)

def train_xgboost(fraud_rate=0.01):

    df = load_data("data/creditcard2023.csv")
    df = simulate_imbalance(df, TARGET, fraud_rate)

    train_df, val_df, test_df = temporal_split(df)

    DROP_COLS = [TARGET, "id"]

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=DROP_COLS)
    y_val = val_df[TARGET]

    # Compute scale_pos_weight
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42
    )

    with mlflow.start_run(run_name=f"xgb_fraudrate_{fraud_rate}"):

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        metrics = {
            "precision": precision_score(y_val, val_preds),
            "recall": recall_score(y_val, val_preds),
            "f1": f1_score(y_val, val_preds),
            "roc_auc": roc_auc_score(y_val, val_probs),
            "pr_auc": average_precision_score(y_val, val_probs)
        }

        mlflow.log_params({
            "model": "XGBoost",
            "fraud_rate": fraud_rate,
            "scale_pos_weight": scale_pos_weight,
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05
        })

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, "artifacts/xgb_model.pkl")

        print(metrics)

if __name__ == "__main__":
    train_logistic(0.01)
    train_logistic(0.05)

    train_xgboost(0.01)
    train_xgboost(0.05)