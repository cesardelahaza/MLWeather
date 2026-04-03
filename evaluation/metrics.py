import json
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

METRICS_PATH = Path("evaluation/metrics.json")

def compute_metrics(y_test, y_pred):
    """
    Model evaluation metrics
    """
    metrics = {}
    
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    metrics['mae'] = round(float(mae), 4)
    metrics['rmse'] = round(float(rmse), 4)
    metrics['r2'] = round(float(r2), 4)


    return metrics

def save_metrics(model_name, y_test, y_pred, extra_info={}):
    """
    Save metrics into a file taking into account the model and extra information
    """
    # Load existent JSON if it exists
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                all_metrics = json.load(f)
        except json.JSONDecodeError:
            print("Warning: metrics.json corrupt")
            all_metrics = {}
    else:
        all_metrics = {}

    all_metrics[model_name] = {
        **compute_metrics(y_test, y_pred),
        **extra_info  # extra information, such as trained_at, features_used, etc.
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)