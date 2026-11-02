from ast import Import

from narwhals import corr
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import false




lables =["a","b","c","d"]
values =[10,20,30,50]

plt.figure(figsize=(5,4))
plt.pie(values, labels=lables)
plt.title(" simple pie chart")
plt.show()

months =["jan","feb","mar","apr","may","jun",]
sales =[12000,15200,12005,12502,15000,18000]

plt.figure(figsize=(7,5))
plt.plot(months,sales,marker='o')
plt.title("monthly sales")
plt.xlabel("months")
plt.ylabel("sales")

plt.show()



days =[1,2,3,4,5,6,7]
daily_sales =[100,200,300,400,500,600,800]

plt.figure(figsize=(7,5))
plt.plot(days,daily_sales,marker='*',linestyle='--',linewidth=1,label='daily sales')
plt.title("daily sales Trend")
plt.xlabel("days")
plt.ylabel("sales")
plt.legend()
plt.show()

categories = [
    "Electronics", "Clothing", "Food", "Books", "Beauty"
]
revenue = [45000, 32000, 27000, 24000, 30000]
orders = [120, 100, 80, 90, 110]
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
axes[0].bar(categories, revenue)
axes[0].set_title("Revenue by Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Revenue")
axes[0].tick_params(axis="x", rotation=30)
 
axes[1].bar(categories, orders)
axes[1].set_title("Orders by Category")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Number of Orders")
axes[1].tick_params(axis="x", rotation=30)
 
plt.tight_layout()
plt.show()


data = {
    "Category": [
        "Electronics", "Clothing", "Food", "Electronics",
        "Clothing", "Food", "Beauty", "Beauty",
        "Electronics", "Food", "Clothing", "Beauty"
    ],
    "Payment Method": [
        "UPI", "Cash", "Card", "UPI",
        "Card", "Cash", "UPI", "Card",
        "Cash", "UPI", "Cash", "Card"
    ],
    "Quantity": [2, 3, 5, 1, 4, 6, 2, 3, 1, 4, 2, 5],
    "Price Per Unit": [
        500, 200, 50, 800, 150, 40,
        300, 250, 1200, 60, 180, 200
    ],
    "Total Spent": [
        1000, 600, 250, 800, 600, 240,
        600, 750, 1200, 240, 360, 1000
    ],
    "Location": [
        "Online", "In-store", "Online", "In-store",
        "Online", "In-store", "Online", "In-store",
        "Online", "In-store", "Online", "In-store"
    ]
}
 
df = pd.DataFrame(data)
 
print("First five rows:")
print(df.head())
 
print("\nDataset information:")
df.info()

plt.figure(figsize=(10,5))
sns.countplot(x='Category', data=df)
plt.title("Number of Transactions by category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Payment Method', hue='Payment Method', legend=False)
plt.title("Number by payment method")
plt.xlabel("payment method")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Total Spent",
    bins=6,
    kde=True
)
plt.title("Distribution of Total Spending")
plt.xlabel("Total Spent")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("correlation heatmap")
plt.show()