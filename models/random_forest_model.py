import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import TimeSeriesSplit

from evaluation.metrics import save_metrics
from datetime import date

import joblib

def random_forest_model(target):
    """
    Build Random Forest Regression model with the historic data as train data
    """
    print("Reading treated historic data...")
    data = pd.read_csv('data/treated_historic_data.csv')

    # Split
    X_train = data.drop(columns=['date', target])
    y_train = data[target]

    cv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
    }

    print("Searching for the best model...")
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30,            
        cv=cv,                
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print("Random forest model acquired...")
    best_model = search.best_estimator_

    print("Saving random forest model parameters information...")
    joblib.dump(best_model, 'models/random_forest_model_info.pkl')
    
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

    rf_model, X_train, _, _ = random_forest_model('temperature_2m_seville')

    X_test, y_test, dates = forecast_data('temperature_2m_seville')

    print('Predicting values with Random Forest Model...')
    y_pred = rf_model.predict(X_test)

    print("Saving predictions to data/predictions/random_forest_model_predictions.csv")
    pred_df = pd.DataFrame({'date': dates,
                            'target': y_test,
                            'y_pred': y_pred,
                            'residuals': y_test - y_pred})
    pred_df.to_csv('data/predictions/random_forest_model_predictions.csv')

    print("Saving metrics of random forest model into evaluation/metrics.json")
    save_metrics(
        model_name='random_forest_model',
        y_test=y_test,
        y_pred=y_pred,
        extra_info={"trained_at": str(date.today()),
                    "importance": rf_model.feature_importances_.tolist(),
                    "features": rf_model.feature_names_in_.tolist()}
    )

if __name__ == "__main__":
    main()