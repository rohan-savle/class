import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons #to create a dummy data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

print("Ready!")
X, _ = make_moons(n_samples=250, noise=0.07, random_state=42)
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-1.5, high=2.5, size=(12, 2))
X = np.vstack([X, outliers])
data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
print("Number of points:", len(data))
print(data.head)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], s=30)
plt.title("Before DBSCAN: unlable points")

plt.xlabel("feacture 1");plt.ylabel("feacture 2")
plt.grid(alpha=0.25);plt.show()

X_scaled = StandardScaler().fit_transform(X)
print("Scaled data shape:",X_scaled.shape)

eps_value = 0.25
min_samples_value = 5

model = DBSCAN(eps=eps_value, min_samples=min_samples_value)
labels = model.fit_predict(X_scaled)
data["Cluster"] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))
print("eps:", eps_value, "| min_samples:", min_samples_value)
print("Clusters found:", n_clusters)
print("Noise points:", n_noise)
print(data["Cluster"].value_counts().sort_index().to_frame("Point count"))

plt.figure(figsize=(8, 5))
for label in sorted(set(labels)):
    mask = labels == label
    if label == -1:
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c="black",
                    marker="x", s=65, label="Noise (-1)")
    else:
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], s=32,
                    label=f"Cluster {label}")
plt.title(f"DBSCAN result: eps={eps_value}, min_samples={min_samples_value}")
plt.xlabel("Scaled Feature 1"); plt.ylabel("Scaled Feature 2")
plt.legend(); plt.grid(alpha=0.2); plt.show()


is_core = np.zeros(len(X_scaled), dtype=bool)
is_core[model.core_sample_indices_] = True
is_noise = labels == -1
is_border = (~is_core) & (~is_noise)

print("Core points:", int(is_core.sum()))
print("Border points:", int(is_border.sum()))
print("Noise points:", int(is_noise.sum()))

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[is_border, 0], X_scaled[is_border, 1],
            c="orange", s=42, label="Border points")
plt.scatter(X_scaled[is_core, 0], X_scaled[is_core, 1],
            c="royalblue", s=35, label="Core points")
plt.scatter(X_scaled[is_noise, 0], X_scaled[is_noise, 1],
            c="black", marker="x", s=65, label="Noise points")
plt.title("DBSCAN point types")
plt.xlabel("Scaled Feature 1"); plt.ylabel("Scaled Feature 2")
plt.legend(); plt.grid(alpha=0.2); plt.show()