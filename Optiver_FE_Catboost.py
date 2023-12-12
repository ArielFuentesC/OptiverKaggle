#Catboost feature engineering for Optiver Kaggle competition

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Load the train dataset
df = pd.read_csv('train.csv')

#Lagged feats
def create_lagged_features(df, features, lags):
    for feature in features:
        for lag in lags:
            df[f"{feature}_lag{lag}"] = df.groupby('stock_id')[feature].shift(lag)
    return df

#Rolling feats

def calculate_rolling_statistics(df, features, windows):
    for feature in features:
        for window in windows:
            df[f"{feature}_rolling_mean_{window}"] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window).mean())
            df[f"{feature}_rolling_std_{window}"] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window).std())
    return df

# Lagged and rolling features as well as windows
lag_features = ['imbalance_size', 'reference_price', 'matched_size']
lags = [10, 30, 50]
rolling_features = ['wap', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
windows = [10, 30, 50]


# Compute feature engineering
df = create_lagged_features(df, lag_features, lags)
df = calculate_rolling_statistics(df, rolling_features, windows)

# Drop rows with missing values
df = df.dropna()

# Feature engineering
df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')

#Train 
columns_to_exclude = ['target', 'date_id', 'time_id', 'row_id']
X = df.drop(columns=columns_to_exclude)

y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a CatBoostRegressor model
model = CatBoostRegressor(loss_function='MAE', verbose=200)

# Define a parameter grid for the grid search
param_grid = {
    'iterations': [300, 500],  
    'depth': [10, 15], 
    'learning_rate': [0.25, 0.35, 0.4]  
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Calculate MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print("Best Parameters:", best_params)
print("Mean Absolute Error (MAE):", mae)