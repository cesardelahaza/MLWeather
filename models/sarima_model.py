import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from evaluation.metrics import save_metrics
from datetime import date

import joblib

def sarima_model(target):
    """
    Build SARIMA Regression model with the historic data as train data
    """
    print("Reading treated historic data...")
    data = pd.read_csv('data/treated_historic_data.csv')

    y_train = data[target]

    model = SARIMAX(y_train,
                    order=(2, 0, 1),
                    seasonal_order=(2, 0, 1, 24))
    
    sarima_fit = model.fit(disp=False)
   
    return sarima_fit, y_train, data['date']

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
    s_model, y_train, _ = sarima_model('temperature_2m_seville')

    X_test, y_test, dates = forecast_data('temperature_2m_seville')

    print('Predicting values with SARIMA Model...')
    n_steps = len(y_test)
    y_pred = list(s_model.forecast(steps=n_steps))

    print("Saving SARIMA model parameters information...")
    s_model.remove_data()
    joblib.dump(s_model, 'models/sarima_model_info.pkl')

    print("Saving predictions to data/predictions/sarima_model_predictions.csv")
    pred_df = pd.DataFrame({'date': dates,
                            'target': y_test,
                            'y_pred': y_pred,
                            'residuals': y_test - y_pred})
    pred_df.to_csv('data/predictions/sarima_model_predictions.csv')

    print("Saving metrics of sarima model into evaluation/metrics.json")
    save_metrics(
        model_name='sarima_model',
        y_test=y_test,
        y_pred=y_pred,
        extra_info={"trained_at": str(date.today()),
                    "importance": '',
                    "features": ''}
    )

if __name__ == "__main__":
    main()