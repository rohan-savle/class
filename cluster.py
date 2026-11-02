
# ============================================
# AGGLOMERATIVE HIERARCHICAL CLUSTERING
# CUSTOMER SEGMENTATION
# ============================================

# 1. IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

print("Libraries imported successfully!")


# 2. LOAD DATASET

df = pd.read_csv("custmer_segmention.csv")

print("\nDataset:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# 3. RENAME COLUMNS IF REQUIRED

rename_map = {
    "Annual_Income": "Annual Income (k$)",
    "Annual Income": "Annual Income (k$)",
    "Spending_Score": "Spending Score (1-100)",
    "Spending Score": "Spending Score (1-100)",
    "Customer_ID": "CustomerID"
}

df = df.rename(columns=rename_map)

# Remove extra spaces
df.columns = df.columns.str.strip()


# 4. CHECK REQUIRED FEATURES

feature = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

missing_features = [
    col for col in feature if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing columns: {missing_features}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# 5. CHECK MISSING VALUES

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset Information:")
df.info()


# 6. SELECT FEATURES

for col in feature:
    df[col] = pd.to_numeric(df[col], errors="coerce")

x = df[feature].copy()

# Handle missing values
x = x.fillna(x.median())

print("\nSelected Features:")
print(x.head())


# 7. VISUALIZE CUSTOMER DATA

plt.figure(figsize=(8, 6))

plt.scatter(
    x["Annual Income (k$)"],
    x["Spending Score (1-100)"],
    alpha=0.75
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Data Before Clustering")
plt.grid(alpha=0.25)

plt.show()


# 8. FEATURE SCALING

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

print("\nData Scaling Completed!")

print("Number of customers:", x_scaled.shape[0])
print("Number of features:", x_scaled.shape[1])


# 9. PAIRWISE DISTANCE EXAMPLE

pairwise_distance_example = [3, 7, 10]

print("\nSingle Linkage Distance:",
      min(pairwise_distance_example))

print("Complete Linkage Distance:",
      max(pairwise_distance_example))


# 10. SINGLE LINKAGE DENDROGRAM

single_linkage = linkage(
    x_scaled,
    method="single"
)

plt.figure(figsize=(12, 6))

dendrogram(single_linkage)

plt.title("Dendrogram - Single Linkage")
plt.xlabel("Customer Index")
plt.ylabel("Distance")

plt.show()


# 11. COMPLETE LINKAGE DENDROGRAM

complete_linkage = linkage(
    x_scaled,
    method="complete"
)

plt.figure(figsize=(12, 6))

dendrogram(complete_linkage)

plt.title("Dendrogram - Complete Linkage")
plt.xlabel("Customer Index")
plt.ylabel("Distance")

plt.show()


# 12. APPLY AGGLOMERATIVE CLUSTERING

N_CLUSTERS = 5

# Single Linkage
single_model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    linkage="single"
)

single_labels = single_model.fit_predict(x_scaled)


# Complete Linkage
complete_model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    linkage="complete"
)

complete_labels = complete_model.fit_predict(x_scaled)


# 13. ADD CLUSTER LABELS

df["single_linkage_Cluster"] = single_labels

df["complete_linkage_cluster"] = complete_labels


# 14. DISPLAY CLUSTER COUNTS

print("\nSingle Linkage Cluster Counts:")

print(
    df["single_linkage_Cluster"]
    .value_counts()
    .sort_index()
    .to_frame("Customers")
)


print("\nComplete Linkage Cluster Counts:")

print(
    df["complete_linkage_cluster"]
    .value_counts()
    .sort_index()
    .to_frame("Customers")
)


# 15. SINGLE LINKAGE VISUALIZATION

plt.figure(figsize=(9, 6))

plt.scatter(
    x["Annual Income (k$)"],
    x["Spending Score (1-100)"],
    c=single_labels,
    cmap="viridis",
    s=60
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation - Single Linkage")
plt.colorbar(label="Cluster")

plt.show()


# 16. COMPLETE LINKAGE VISUALIZATION

plt.figure(figsize=(9, 6))

plt.scatter(
    x["Annual Income (k$)"],
    x["Spending Score (1-100)"],
    c=complete_labels,
    cmap="viridis",
    s=60
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation - Complete Linkage")
plt.colorbar(label="Cluster")

plt.show()


# 17. SAVE FINAL DATASET

df.to_csv("clustered_customer_data.csv", index=False)

print("\nFinal dataset saved successfully!")

print("\nAgglomerative Clustering Completed Successfully!")
