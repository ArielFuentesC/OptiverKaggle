import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load your dataset (replace 'train.csv' with your actual dataset)
df = pd.read_csv('train.csv')

# Drop rows with missing values
df = df.dropna()

# Feature engineering
df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')

# Define features and target
X = df[['seconds_in_bucket', 'imbalance_size', 'imbalance_buy_sell_flag', 'reference_price', 'matched_size', 'far_price',
        'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'imb_s1', 'imb_s2']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming 'date_id' represents time, set it as the index
df['date_id'] = pd.to_datetime(df['date_id'])
df.set_index('date_id', inplace=True)

# Create a time series for the 'target' variable
time_series = df['target']

# Perform Dickey-Fuller test to check stationarity
result = adfuller(time_series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, the series is not stationary; we need differencing
if result[1] > 0.05:
    # Perform differencing (make it stationary)
    diff_time_series = time_series.diff().dropna()

    # Re-run the Dickey-Fuller test
    result = adfuller(diff_time_series)
    print('ADF Statistic after differencing:', result[0])
    print('p-value after differencing:', result[1])

# Now, you can fit an ARIMA model
# Assuming order=(p, d, q), where p, d, and q are parameters
p, d, q = 1, 1, 1  # You can adjust these values

model = ARIMA(time_series, order=(p, d, q))
model_fit = model.fit(disp=0)

# Make predictions
predictions = model_fit.forecast(steps=len(X_test))[0]

# Calculate Mean Absolute Error (MAE) for model evaluation
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error (MAE):', mae)

# Plot the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='red')
plt.legend()
plt.title('ARIMA Model - Actual vs. Predicted')
plt.show()