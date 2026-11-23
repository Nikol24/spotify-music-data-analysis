
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "spotify_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_FILE)

# Select audio features
features = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

df = df.dropna(subset=features).copy()

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Evaluate cluster counts using a sample
rng = np.random.default_rng(42)

sample_size = min(5000, len(df))
sample_idx = rng.choice(
    len(df),
    size=sample_size,
    replace=False
)

X_sample = X[sample_idx]
scores = []

for k in range(2, 9):
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        batch_size=1024
    )

    labels = model.fit_predict(X_sample)

    score = silhouette_score(
        X_sample,
        labels,
        sample_size=min(2000, sample_size),
        random_state=42
    )

    scores.append((k, score))
    print(f"Clusters: {k}, Silhouette: {score:.4f}")

# Select the best number of clusters
best_k = max(scores, key=lambda x: x[1])[0]

print("\nSelected number of clusters:", best_k)

# Fit final model
kmeans = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
    batch_size=1024
)

df["cluster"] = kmeans.fit_predict(X)

# Cluster summary
cluster_summary = df.groupby("cluster")[features].mean()
cluster_summary["track_count"] = df.groupby("cluster").size()

print("\nCluster summary:")
print(cluster_summary.round(3))

# Export summary statistics
cluster_summary.to_csv(
    RESULTS_DIR / "cluster_summary.csv"
)

pd.DataFrame(
    scores,
    columns=["n_clusters", "silhouette_score"]
).to_csv(
    RESULTS_DIR / "silhouette_scores.csv",
    index=False
)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_idx = rng.choice(
    len(df),
    size=min(10000, len(df)),
    replace=False
)

plt.figure(figsize=(11, 7))

scatter = plt.scatter(
    X_pca[plot_idx, 0],
    X_pca[plot_idx, 1],
    c=df["cluster"].to_numpy()[plot_idx],
    cmap="tab10",
    s=8,
    alpha=0.5
)

plt.colorbar(scatter, label="Cluster")
plt.title("Spotify Music Clustering — PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "spotify_clusters_pca.png",
    dpi=300
)

plt.close()

print("\nClustering completed successfully.")
