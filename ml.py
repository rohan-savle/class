import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df =pd.read_csv("real estate.csv")
print(df.head())
print(df.columns.tolist())


print("First five rows:")
print(df.head())
 
print("\nDataset shape:")
print(df.shape)
 
print("\nColumn names:")
print(df.columns.tolist())
 
print("\nDataset information:")
df.info()
 
print("\nMissing values:")
print(df.isnull().sum())
 
print("\nDuplicate rows:")
print(df.duplicated().sum())






plt.figure(figsize=(10, 7))
sns.histplot(df["lakh"], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("House Price per Unit Area")
plt.ylabel("Frequency")
plt.show()