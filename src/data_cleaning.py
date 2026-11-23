
import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "dataset.csv"
OUTPUT_FILE = DATA_DIR / "spotify_cleaned.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Remove unnecessary index column
df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)

# Remove missing essential information
df = df.dropna(
    subset=[
        "track_id",
        "artists",
        "track_name",
        "album_name"
    ]
).copy()

# Remove exact duplicate rows
df = df.drop_duplicates()

# Convert numerical columns
numeric_columns = [
    "popularity",
    "duration_ms",
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

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(subset=numeric_columns)

# Validate values
df = df[
    df["popularity"].between(0, 100)
    & (df["duration_ms"] > 0)
    & (df["tempo"] >= 0)
].copy()

# Validate audio features
audio_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

for feature in audio_features:
    df = df[df[feature].between(0, 1)]

# Convert milliseconds to minutes
df["duration_min"] = df["duration_ms"] / 60000

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned dataset:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nData cleaning completed successfully.")
print("Saved to:", OUTPUT_FILE)
