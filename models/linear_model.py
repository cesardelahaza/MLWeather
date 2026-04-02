import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluation.metrics import save_metrics
from datetime import date

def linear_model(target):
    """
    Build linear model with the historic data as train data
    """
    print("Reading treated historic data...")
    data = pd.read_csv('data/treated_historic_data.csv')

    # Split
    X_train = data.drop(columns=['date', target])
    y_train = data[target]
    
    print("Building the model with the data...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    return lr_model, X_train, y_train, data['date']

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
    lr_model, X_train, _, _ = linear_model("temperature_2m_seville")

    X_test, y_test, dates = forecast_data("temperature_2m_seville")

    print("Predicting values...")
    y_pred = lr_model.predict(X_test)

    print("Saving predictions to data/predictions/linear_model_predictions.csv")
    pred_df = pd.DataFrame({'date': dates,
                            'target': y_test,
                            'y_pred': y_pred,
                            'residuals': y_test - y_pred})
    pred_df.to_csv('data/predictions/linear_model_predictions.csv')

    print("Saving metrics of linear model")

    save_metrics(
        model_name='linear_model',
        y_test=y_test,
        y_pred=y_pred,
        extra_info={"trained_at": str(date.today()),
                    "coefficients": lr_model.coef_.tolist(),
                    "features": X_train.columns.tolist()}
    )


if __name__ == "__main__":
    main()

