# Air Passengers Time Series Analysis 

## Overview

This document provides detailed documentation for the time series analysis conducted on the "Air Passengers" dataset, which contains monthly totals of international airline passengers from 1949 to 1960. The analysis utilizes various time series techniques, including decomposition, stationarity testing, and SARIMA modeling for forecasting future passenger numbers.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Time Series Decomposition](#time-series-decomposition)
4. [Stationarity Analysis](#stationarity-analysis)
5. [Model Parameter Selection](#model-parameter-selection)
6. [SARIMA Model Implementation](#sarima-model-implementation)
7. [Forecasting](#forecasting)
8. [Appendix: Code Explanation](#appendix-code-explanation)

## Data Preparation

The analysis begins with importing the Air Passengers dataset, which contains monthly passenger counts in international air travel. The dataset is loaded from a CSV file, and the 'Month' column is set as the index.

```python
df = pd.read_csv('AirPassengers.csv')
df.set_index('Month', inplace=True)
```

## Exploratory Data Analysis

The first step in analyzing time series data is visualizing the raw data to identify patterns, trends, and potential seasonality.

![Passengers Over Time](https://via.placeholder.com/800x200)

**Observations:**
- The data shows a clear upward trend over time, indicating growth in air travel.
- There are regular seasonal patterns visible, with peaks occurring at consistent intervals.
- The magnitude of seasonal fluctuations increases with the level of the series, suggesting a multiplicative seasonal pattern.

## Time Series Decomposition

Time series decomposition breaks down the original series into its constituent components: trend, seasonality, and residuals. A multiplicative model is used because the seasonal variations increase with the level of the series.

```python
result = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
```

![Time Series Decomposition](https://via.placeholder.com/800x600)

**Component Analysis:**

1. **Trend Component**: Shows the long-term progression of the series, clearly illustrating the overall growth in air travel over time.

2. **Seasonal Component**: Reveals regular patterns that repeat every 12 months. The seasonal pattern remains consistent, showing higher passenger counts during summer months and lower counts during winter months.

3. **Residual Component**: Represents the variations that cannot be attributed to trend or seasonality. Ideally, residuals should be random noise with no discernible pattern.

## Stationarity Analysis

Stationarity is a crucial assumption for many time series models. A stationary series has constant mean, variance, and autocorrelation structure over time. The Augmented Dickey-Fuller (ADF) test is used to check for stationarity.

```python
result = adfuller(df['#Passengers'], autolag='AIC')
```

**ADF Test Results:**
- Original Series: Non-stationary (p-value > 0.05)
- First-order Differencing: Stationary (p-value < 0.05)
- Second-order Differencing: Strongly stationary (p-value << 0.05)

![Differencing Results](https://via.placeholder.com/800x600)

**Interpretation:**
- The original series shows a clear trend and is non-stationary.
- After first-order differencing, the series becomes stationary, indicating that d=1 is an appropriate parameter for our ARIMA model.
- Second-order differencing might be excessive and could lead to overfitting.

## Model Parameter Selection

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are used to determine the appropriate orders for the ARIMA model.

![ACF and PACF Plots](https://via.placeholder.com/800x400)

**Parameter Selection:**

- **p (AR order)**: 2 - Based on significant spikes in the PACF plot
- **d (Differencing order)**: 1 - Based on stationarity tests
- **q (MA order)**: 1 - Based on significant spikes in the ACF plot
- **P (Seasonal AR order)**: 1
- **D (Seasonal differencing)**: 0 - No seasonal differencing needed
- **Q (Seasonal MA order)**: 3
- **Seasonal period**: 12 (monthly data)

## SARIMA Model Implementation

The Seasonal Autoregressive Integrated Moving Average (SARIMA) model is implemented with the parameters determined in the previous step.

```python
model = SARIMAX(df['#Passengers'], order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
fitted_model = model.fit()
```

**Model Summary:**

The model summary provides various diagnostics including log-likelihood, AIC, BIC, and parameter estimates along with their standard errors and significance levels. A good model should have:

- Low AIC and BIC values
- Significant parameter estimates (p-values < 0.05)
- Residuals that resemble white noise

## Forecasting

The fitted SARIMA model is used to forecast passenger numbers for the next 24 months (2 years).

```python
forecast_steps = 24
forecast = fitted_model.get_forecast(steps=forecast_steps)
```

![Forecast Results](https://via.placeholder.com/800x400)

**Forecast Analysis:**

- The forecast shows continued growth in passenger numbers, following the established trend.
- The seasonal patterns are preserved in the forecast.
- The confidence interval widens as we forecast further into the future, reflecting increasing uncertainty.

## Appendix: Code Explanation

### Libraries Used

```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import statsmodels.api as sm     # Statistical models
from statsmodels.tsa.seasonal import seasonal_decompose  # Time series decomposition
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns            # Enhanced plotting
```

### Data Loading and Exploration

```python
df = pd.read_csv('AirPassengers.csv')
df.set_index('Month', inplace=True)
df.head()
```

This section loads the dataset and sets the 'Month' column as the index for time series analysis.

### Time Series Visualization

```python
plt.figure(figsize=(20, 5))
plt.plot(df.index, df['#Passengers'], label='#Passengers')
```

This code creates a line plot of passenger counts over time to visualize trends and patterns.

### Seasonal Decomposition

```python
result = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
```

The `seasonal_decompose` function breaks down the time series into trend, seasonality, and residual components using a multiplicative model with a period of 12 months.

### Stationarity Testing

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['#Passengers'], autolag='AIC')
```

The Augmented Dickey-Fuller test checks for stationarity in the time series data. The test is applied to the original series and its differenced versions.

### ACF and PACF Analysis

```python
sm.graphics.tsa.plot_acf(df.diff().dropna(), lags=40, ax=ax[0])
sm.graphics.tsa.plot_pacf(df.diff().dropna(), lags=40, ax=ax[1])
```

These functions create autocorrelation and partial autocorrelation plots to help determine appropriate AR and MA orders.

### SARIMA Model

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(df['#Passengers'], order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
```

SARIMAX is used to create a seasonal ARIMA model with the specified parameters.

### Forecasting Implementation

```python
forecast = fitted_model.get_forecast(steps=forecast_steps)
forecast_df = pd.DataFrame({
    "Forecast": list(forecast.predicted_mean),
    "Lower CI": list(forecast.conf_int().iloc[:, 0]),
    "Upper CI": list(forecast.conf_int().iloc[:, 1])
}, index=forecast_index)
```

This code generates forecasts for future time periods and creates a DataFrame with point forecasts and confidence intervals.
