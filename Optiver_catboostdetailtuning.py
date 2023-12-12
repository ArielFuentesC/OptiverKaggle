import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

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

# Define your parameter grid for the grid search as a list of dictionaries
param_grid = [
    {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 1, 'border_count': 32},
    {'iterations': 1000, 'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'border_count': 64},
    {'iterations': 1500, 'depth': 10, 'learning_rate': 0.2, 'l2_leaf_reg': 5, 'border_count': 128}
]

# Create a CatBoost dataset
train_data = Pool(X_train, y_train)
test_data = Pool(X_test, y_test)  # Define the test dataset

# Define early stopping criteria
early_stopping_rounds = 20  # Number of rounds with no significant improvement
od_type = 'Iter'  # Track by the number of iterations
od_wait = early_stopping_rounds

best_mae = float('inf')  # Initialize the best MAE with a large value
best_params = {}  # Initialize the best parameters
best_model = None  # Initialize the best model

# Iterate through the parameter grid
for params in param_grid:
    print(f"Training with params: {params}")

    # Initialize a new CatBoostRegressor with the current parameters
    model = CatBoostRegressor(
        loss_function='MAE',
        verbose=200,  
        od_type=od_type,
        od_wait=od_wait,
        **params  # Pass the parameters from the grid
    )

    # Fit the model with early stopping and the test dataset for evaluation
    model.fit(train_data, eval_set=test_data, use_best_model=True)

    # Check if this model achieved a better MAE
    if 'validation' in model.get_best_score() and 'MAE' in model.get_best_score()['validation']:
        validation_mae = model.get_best_score()['validation']['MAE']
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_params = params
            best_model = model

# Print the best parameters and MAE achieved
print(f"Best Parameters: {best_params}")
print(f"Best MAE: {best_mae}")

# Make predictions using the best model
predictions = best_model.predict(test_data)

output_df = pd.DataFrame({
    'row_id': X_test.index,  
    'target': predictions
})

# Save the predictions to a CSV file
output_df.to_csv('submission.csv', index=False)