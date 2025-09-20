# utils.py
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob[:,1]),
        "AUPR": average_precision_score(y_true, y_prob[:,1])
    }
