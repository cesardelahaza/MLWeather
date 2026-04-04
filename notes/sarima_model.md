# SARIMA

## Time Series

A time series is a sequence of observations indexed in time order. In forecasting, the goal is to predict future values based on past observations.

## Stationarity

A time series is stationary if its statistical properties (mean, variance, autocorrelation) do not change over time. Many classical forecasting models require stationarity. A common way to achieve it is **differencing**: subtracting consecutive observations to remove trends.

## ARIMA

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for time series forecasting. It has three components:

- **AR (AutoRegressive):** the value at time $t$ depends on its own past values.
- **I (Integrated):** the number of differencing operations applied to make the series stationary.
- **MA (Moving Average):** the value at time $t$ depends on past forecast errors.

ARIMA is written as ARIMA$(p, d, q)$, where $p$ is the AR order, $d$ the differencing order, and $q$ the MA order.

## SARIMA

SARIMA (Seasonal ARIMA) extends ARIMA by adding seasonal components. Temperature data, for example, has clear daily and yearly cycles that ARIMA alone cannot capture.

SARIMA is written as ARIMA $(p, d, q)(P, D, Q)_s$, where the second set of parameters mirrors the first but operates at the seasonal lag $s$ (e.g., $s=24$ for hourly data with a daily cycle).

### SARIMA for Regression

SARIMA is a purely statistical model: it does not use external features, only the past values of the series itself. It is interpretable and well-suited for univariate forecasting with strong seasonal patterns.

### Some configurable Hyperparameters

| Hyperparameter | Description |
|---|---|
| `p` | AR order: number of past values used |
| `d` | Differencing order: how many times the series is differenced |
| `q` | MA order: number of past forecast errors used |
| `P, D, Q` | Seasonal equivalents of `p`, `d`, `q` |
| `s` | Seasonal period (e.g., 24 for hourly data with daily seasonality) |