import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from evaluation.metrics import save_metrics
from datetime import date

def xgboost_model(target):
    """
    Build XGBoost Regression model with the historic data as train data
    """
    print("Reading treated historic data...")
    data = pd.read_csv('data/treated_historic_data.csv')

    # Split
    X_train = data.drop(columns=['date', target])
    y_train = data[target]

    model = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    print("Searching for the best model...")
    gs = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
    )

    gs.fit(X_train, y_train)
    print("XGBoost model acquired...")
    best_model = gs.best_estimator_

    print("Saving XGBoost model information...")
    best_model.save_model('models/xgboost_model_info.json')
    
    return best_model, X_train, y_train, data['date']

def forecast_data(target):
    """
    Reads forecast data as the test data
    """
    print("Reading treated forecast data...")
    data = pd.read_csv('data/treated_forecast_data.csv') 

    X_test = data.drop(columns=['date', target])
    y_test = data[target]

    return X_test, y_test, data['date']

def main():
    xgb_model, _, _, _ = xgboost_model('temperature_2m_seville')

    X_test, y_test, dates = forecast_data('temperature_2m_seville')

    print("Predicting values...")
    y_pred = xgb_model.predict(X_test)

    print("Saving predictions to data/predictions/xgboost_model_predictions.csv")
    pred_df = pd.DataFrame({'date': dates,
                            'target': y_test,
                            'y_pred': y_pred,
                            'residuals': y_test - y_pred})
    pred_df.to_csv('data/predictions/xgboost_model_predictions.csv')

    print("Saving metrics of xgboost model into evaluation/metrics.json")
    save_metrics(
        model_name='xgboost_model',
        y_test=y_test,
        y_pred=y_pred,
        extra_info={"trained_at": str(date.today()),
                    "importance": xgb_model.feature_importances_.tolist(),
                    "features": xgb_model.feature_names_in_.tolist()}
    )

if __name__ == "__main__":
    main()