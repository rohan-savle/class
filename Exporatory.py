import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import column

df = pd.read_csv('retail_store_sales.csv')

df.head()
 
# Display last five rows

df.tail()
 
# Total rows

df.shape[0]
 
# Total columns

df.shape[1]
 
# Column names

df.columns
 
# Data types

df.dtypes
 
# Dataset information

df.info()
 
# Statistical summary of numerical columns

df.describe()
 
# Statistical summary of categorical columns

df.describe(include='object')


missing = df.isnull().sum()
print("missing")
print(missing)

missing_percentage = (missing / len(df)) * 100
missing_table = pd.DataFrame({'Missing Count': missing, 'Missing Percentage': (round(missing_percentage, 2))})



plt.figure(figsize=(10,5))
sns.barplot(x=missing.index, y=missing.values)
plt.xticks(rotation=45)
plt.ylabel('Number of Missing Values')
plt.xlabel('Missing Values by Column')
plt.title('Missing Values by Column')
plt.tight_layout()
plt.show()


for column in df.columns:
    print("\n",column)
    print("Unique Values:",df[column].nunique())



    print(df['Payment Method'].value_counts())



print('retail_store_sales.csv'.head(10))
 
# Plot top 10 items by revenue

plt.figure(figsize=(7, 5))
 
sns.barplot(

    x='retail_store_sales.csv'.head(10).values,

    y='retail_store_sales.csv'.head(10).index

)
 
plt.title("Top 10 Items by Revenue")

plt.xlabel("Total Revenue")

plt.ylabel("Item")

plt.tight_layout()

plt.show()
 
# ============================================================

# 19. QUANTITY ANALYSIS

# ============================================================
 
# Calculate total quantity sold for each item

item_quantity = (

    df.groupby("Item")["Quantity"]

    .sum()

    .sort_values(ascending=False)

)
 
# Display top five items by quantity sold

print(item_quantity.head(5))
 
# ============================================================

# 20. DISCOUNT ANALYSIS

# ============================================================
 
# Compare transaction count, total revenue, and average revenue

# between transactions with and without discounts

discount_analysis = (

    df.groupby("Discount Applied")["Total Spent"]

    .agg(['count', 'sum', 'mean'])

)
 
print(discount_analysis)
 
# Plot average Total Spent by discount status

plt.figure(figsize=(7, 5))
 
sns.barplot(

    data=df,

    x="Discount Applied",

    y="Total Spent"

)
 
plt.title("Total Spent by Discount Applied")

plt.xlabel("Discount Applied")

plt.ylabel("Total Spent")

plt.tight_layout()

plt.show()
 
# ============================================================

# 21. FINAL CLEANED DATASET

# ============================================================
 
print("Final dataset shape:", df.shape)
 
print("\nRemaining missing values:")

print(df.isnull().sum())
 
print("\nRemaining duplicate records:")

print(df.duplicated().sum())
 
# Display final dataset

df.head()
 