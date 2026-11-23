
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "spotify_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# 1. Top 10 genres
genre_counts = df["track_genre"].value_counts().head(10)

plt.figure(figsize=(10, 6))
genre_counts.sort_values().plot(kind="barh")
plt.title("Top 10 Spotify Genres by Record Count")
plt.xlabel("Number of Records")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "top_10_genres.png", dpi=300)
plt.close()

# 2. Popularity distribution
plt.figure(figsize=(10, 6))
plt.hist(df["popularity"], bins=30, edgecolor="black")
plt.title("Spotify Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "popularity_distribution.png", dpi=300)
plt.close()

# 3. Average popularity by genre
genre_popularity = (
    df.groupby("track_genre")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
genre_popularity.sort_values().plot(kind="barh")
plt.title("Top 10 Genres by Average Popularity")
plt.xlabel("Average Popularity")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "genre_popularity.png", dpi=300)
plt.close()

# 4. Energy vs popularity
plt.figure(figsize=(10, 6))
plt.scatter(
    df["energy"],
    df["popularity"],
    alpha=0.1,
    s=5
)
plt.title("Energy vs Popularity")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "energy_vs_popularity.png", dpi=300)
plt.close()

# 5. Danceability vs popularity
plt.figure(figsize=(10, 6))
plt.scatter(
    df["danceability"],
    df["popularity"],
    alpha=0.1,
    s=5
)
plt.title("Danceability vs Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "danceability_vs_popularity.png", dpi=300)
plt.close()

# 6. Correlation matrix
features = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

correlation = df[features].corr()

plt.figure(figsize=(11, 9))
plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.colorbar(label="Pearson Correlation")
plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(features)), features)
plt.title("Spotify Audio Features Correlation Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=300)
plt.close()

# Save statistical tables
genre_counts.to_csv(
    RESULTS_DIR / "top_10_genres.csv",
    header=["record_count"]
)
genre_popularity.to_csv(
    RESULTS_DIR / "genre_popularity.csv",
    header=["average_popularity"]
)
correlation.to_csv(
    RESULTS_DIR / "correlation_matrix.csv"
)

print("EDA completed successfully.")
print("Results saved in:", RESULTS_DIR)
