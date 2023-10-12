import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

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

# Create a CatBoostRegressor model
model = CatBoostRegressor(loss_function='MAE', verbose=200)

# Define a parameter grid for the grid search
param_grid = {
    'iterations': [500, 1000],  
    'depth': [4, 6, 8], 
    'learning_rate': [0.01, 0.1, 0.2]  
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