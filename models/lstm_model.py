import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from evaluation.metrics import save_metrics
from datetime import date

def lstm_model(target):
    """
    Build LSTM Regression model with the historic data as train data
    """
    print("reading treated historic data...")
    h_data = pd.read_csv('data/treated_historic_data.csv')

    print("Reading treated forecast data...")
    f_data = pd.read_csv('data/treated_forecast_data.csv') 

    # Split
    X_train = h_data.drop(columns=['date', target])
    y_train = h_data[target]

    X_test = f_data.drop(columns=['date', target])
    y_test = f_data[target]

    # Scaling data
    print("Scaling the data...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_s  = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Reshape data into 3D
    n_features = X_train_s.shape[1]

    X_train_3d = X_train_s.reshape((X_train_s.shape[0], 1, n_features))
    X_test_3d  = X_test_s.reshape((X_test_s.shape[0], 1, n_features))

    # Write model
    print("Training LSTM model...")
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(1, n_features)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train model
    history = model.fit(
        X_train_3d, y_train_s,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    print("Saving LSTM model information...")
    model.save('models/lstm_model_info.keras')


    # Predict and unscale
    print("Predicting values for LSTM model...")
    y_pred_s = model.predict(X_test_3d)
    y_pred   = scaler_y.inverse_transform(y_pred_s).reshape(-1,)
    y_real   = scaler_y.inverse_transform(y_test_s).reshape(-1,)

    return y_real, y_pred, f_data['date']

def main():
    y_test, y_pred, dates = lstm_model('temperature_2m_seville')

    print("Saving predictions to data/predictions/lstm_model_predictions.csv")
    pred_df = pd.DataFrame({'date': dates,
                            'target': y_test,
                            'y_pred': y_pred,
                            'residuals': y_test - y_pred})
    pred_df.to_csv('data/predictions/lstm_model_predictions.csv')

    print("Saving metrics of lstm model into evaluation/metrics.json")
    save_metrics(
        model_name='lstm_model',
        y_test=y_test,
        y_pred=y_pred,
        extra_info={"trained_at": str(date.today()),
                    "importance": '',
                    "features": ''}
    )

if __name__ == "__main__":
    main()