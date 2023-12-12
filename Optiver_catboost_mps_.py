import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import time
import matplotlib.pyplot as plt

# Load the train dataset
df = pd.read_csv('train.csv')

#df = df[df['stock_id'] == 0]

# Define a function to create lagged features
def create_lagged_features(df, features, lags):
    for feature in features:
        for lag in lags:
            df[f"{feature}_lag{lag}"] = df.groupby('stock_id')[feature].shift(lag)
    
    return df

# Define a function to calculate rolling statistics
def calculate_rolling_statistics(df, features, windows):
    for feature in features:
        for window in windows:
            df[f"{feature}_rolling_mean_{window}"] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window).mean())
            df[f"{feature}_rolling_std_{window}"] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window).std())
    
    return df

# Function for feature engineering
def feat_engin(df, lag_features, roll_features, lags, windows):
    df = df.dropna()
    columns_to_exclude = ['date_id', 'time_id', 'row_id'] #Move to the front
    df = df.drop(columns=columns_to_exclude)
    df = create_lagged_features(df, lag_features, lags)
    df = calculate_rolling_statistics(df, roll_features, windows)
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
    columns_to_exclude = ['date_id', 'time_id', 'row_id']
    
    return df

# Lagged and rolling features as well as windows
lag_features = ['imbalance_size', 'reference_price', 'matched_size']
lags = [10, 30, 50]
rolling_features = ['wap', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
windows = [10, 30, 50]

# Perform feature engineering on the entire dataset
df = feat_engin(df, lag_features, rolling_features, lags, windows)

# Create a dictionary to store models per stock
models = {}
training_info = {}
learning_scores = {}

# Define a parameter grid for the grid search specific to the stock
param_grid = {
    'iterations': [200, 300, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.15, 0.23, 0.3]
}

# Initialize variables to control subplot layout
rows = 2  # Number of rows in the grid
cols = 5  # Number of columns in the grid
subplot_index = 0

# Iterate over each stock
for stock_id in df['stock_id'].unique():
    # Get data for the current stock
    stock_data = df[df['stock_id'] == stock_id]
    
    # Define X and y for the stock
    y = stock_data['target']
    X = stock_data.drop(columns=['target'])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Create a CatBoostRegressor model for the stock
    model = CatBoostRegressor(loss_function='MAE', verbose=20)

     # Record the start time
    start_time = time.time()   
    
    # Perform grid search with cross-validation for the stock
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model for the stock
    best_model = grid_search.best_estimator_

    # Record the end time
    end_time = time.time()
    
    # Calculate the training time
    training_time = end_time - start_time

    # Store the best model in the dictionary
    models[stock_id] = best_model
    
    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)
    
    # Calculate MAE (Mean Absolute Error) for the stock
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Stock {stock_id}: Mean Absolute Error (MAE): {mae}")

    # Save the training information (params and MAE)
    training_info[stock_id] = {
        'params': grid_search.best_params_,
        'mae': mae,
        'training_time': training_time,
        'training_curve': -grid_search.cv_results_['mean_test_score']
    }
    # Save learning scores for each set of parameters
    learning_scores[stock_id] = -grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']

    # Plot the learning curve and save it as an image
    if subplot_index == 0:
        fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
    
    ax = axs[subplot_index // cols, subplot_index % cols]
    ax.plot(range(len(learning_scores[stock_id])), learning_scores[stock_id], marker='o', label=f'Stock {stock_id}')
    ax.set_title(f'Stock {stock_id}')
    ax.set_xlabel('Parameter Combinations')
    ax.set_ylabel('Negative MAE')

    subplot_index += 1

    # Display 10 learning curves per image
    if subplot_index == rows * cols:
        subplot_index = 0
        plt.tight_layout()
        plt.savefig(f'stocks_{stock_id // 10}_learning_curves.png')
        plt.close(fig)

# Save any remaining plots
if subplot_index > 0:
    plt.tight_layout()
    plt.savefig(f'stocks_{stock_id // 10 + 1}_learning_curves.png')
    plt.close()


