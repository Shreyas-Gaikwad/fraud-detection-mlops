import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def threshold_metrics(y_true, y_probs, thresholds):
    records = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        fpr = fp / (fp + tn)

        records.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "fpr": fpr
        })

    return pd.DataFrame(records)