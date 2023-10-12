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

# Create a CatBoost dataset
train_data = Pool(X_train, y_train)
test_data = Pool(X_test, y_test)  # Define the test dataset


# CatBoostRegressor with the best parameters
model = CatBoostRegressor(
    loss_function='MAE',
    verbose=20,
    iterations = 1500, depth = 10, learning_rate = 0.2, l2_leaf_reg = 5, border_count = 128
)

# Fit the model with early stopping and the test dataset for evaluation
model.fit(train_data, eval_set=test_data, use_best_model=True)

# Make predictions using the best model
predictions = model.predict(test_data)