# MLWeather

Project to predict the temperature in Seville. The structure of the project is (not in its final form):

```
MLWeather/
│
├── README.md
├── requirements.txt                # Libraries and versions
├── .gitignore
├── .python-version
|
├── app.py                          # Streamlit app
|
├── data/
│   ├── historic_data.csv           # Historic data of weather
│   └── forecast_data.csv           # Forecast data
|
├── notebooks/                      # Study of data and models...
│   ├── forecast_analysis.ipynb     # Analysis of forecast data
|   └── historic_analysis.ipynb     # Analysis of historic data
|
├── src/
|   ├── data_treatment.py           # Reading, treatment, preprocessing... 
|   ├── forecast_data.py            # Forecast data collection
|   └── historic_data.py            # Historic data collection
|
├── models/
│   ├── linear_model.py             # Linear Regression + lags
│   ├── xgboost_model.py            # XGBoost
│   ├── random_forest.py            # Random Forest
│   ├── sarima_model.py             # SARIMA
│   └── lstm_model.py               # LSTM
│
├── evaluation/
│   └── metrics.py                  # MAE, RMSE...
│
└── ui/
    ├── sidebar.py                  # Model selection, parameters...
    ├── charts.py                   # Graphics
    └── comparison.py               # Comparative tables
```

## Run Streamlit

```
streamlit run app.py
```

## Getting the Data

The Data is getting from [Open-Meteo](https://open-meteo.com/). In the [_Available APIs_](https://open-meteo.com/en/features#available_apis) section we can get:

* [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api): This data is to build the model. The data I will use here will come from different places around Seville, which will be the place I will predict.
* [Forecast API](https://open-meteo.com/en/docs): This data is to predict the weather in Seville based on different locations near it.



