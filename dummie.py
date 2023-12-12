import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train dataset
df = pd.read_csv('train.csv')

tst = df[df['stock_id'] == 0]
tst.to_csv('tst')
