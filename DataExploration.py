#Data Exploration for Optiver Kaggle competition

#! C:\Users\ariel\Desktop\Optiver\generalenv\Scripts\python.exe

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train dataset
df = pd.read_csv('train.csv')

# Basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())

# Summary statistics for numerical columns
print(df.describe())

# Visualize the distribution of the target variable (60-second future move)
plt.figure(figsize=(10, 6))
plt.hist(df['target'], bins=50, edgecolor='k')
plt.xlabel('Target (Basis Points)')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable')
plt.show()

# Visualize the correlation matrix of numerical features
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.show()

# Visualize a scatter plot of 'imbalance_size' vs. 'target'
plt.figure(figsize=(10, 6))
plt.scatter(df['imbalance_size'], df['target'], alpha=0.5)
plt.xlabel('Imbalance Size (USD)')
plt.ylabel('Target (Basis Points)')
plt.title('Scatter Plot: Imbalance Size vs. Target')
plt.show()

# Visualize a scatter plot of 'wap' vs. 'target'
plt.figure(figsize=(10, 6))
plt.scatter(df['wap'], df['target'], alpha=0.5)
plt.xlabel('Weighted Average Price (WAP)')
plt.ylabel('Target (Basis Points)')
plt.title('Scatter Plot: WAP vs. Target')
plt.show()

# Check for missing values
missing_values = df.isnull().sum()

# Print columns with missing values and their respective counts
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Visualize missing values using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()
