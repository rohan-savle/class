import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

df = pd.read_csv("customers.csv")
print(df.head())

features =[
    "Age",
    "Annual Income",
    "Spending Score"
]

X = df[features].dropna()
print(X.head)


scaler =MinMaxScaler()
X_scaled =scaler.fit_transform(X)


print("Mean of scaled features:")

print(X_scaled.mean(axis=0))
 
print("\nStandard deviation of scaled features:")

print(X_scaled.std(axis=0))
 
print("\nScaled data:")

print(X_scaled)

wcss = []

k_values = range (1,11)

for k in k_values :
    model =KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=1
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)
    print("\nWCSS values.")
    print(wcss)

plt.figure(figsize=(9,6))  
plt.plot(
    k_values,
    wcss,
    marker="o"

)
plt.xlabel("Number of Cluster(K)")
plt.ylabel("WCSS/inertia")
plt.title("Elbol Method for Optimal K")
plt.xticks(list(k_values))
plt.grid(True,alpha=0.3)
plt.show() 

optimal_k =3
final_model = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    n_init=10,
    random_state=1
)

labels = final_model.fit_predict(X_scaled)
 
print("\nCluster labels:")

print(labels)


df_clustered =df.loc[X.index].copy()
df_clustered["Cluster"] = labels
print("/nClustered dataset:")
print(df_clustered.head(20))

print("\nFinal model inertia:")

print(final_model.inertia_)

sil_score = silhouette_score(X_scaled,labels)
print("/nSilhouette Score.")
print(sil_score)

plt.figure(figsize=(9,6))

plt.scatter(

    df_clustered["Annual Income"],

    df_clustered["Spending Score"],

    c=labels,

    cmap="viridis"

)
 
plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.title("Customer Segmentation with K-Means")

plt.colorbar(label="Cluster")

plt.show()

print("\nMinimum feature values in each cluster:")

print(df_clustered.groupby("Cluster")[features].min())