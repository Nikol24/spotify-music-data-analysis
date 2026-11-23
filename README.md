# Spotify Music Data Analysis & Machine Learning

**Data Science Portfolio Project | Python | Pandas | Matplotlib | Scikit-learn**

## Project Overview

This project explores Spotify music data through data cleaning, exploratory data analysis (EDA), statistical visualization, and unsupervised machine learning.

The dataset contains approximately 114,000 track records spanning 114 music genres.

The main objective is to investigate relationships between musical characteristics and popularity, identify patterns across genres, and group tracks according to similarities in their audio features.

## Project Objectives

- Prepare and clean a real-world music dataset.
- Explore the distribution of track popularity.
- Identify the most represented music genres.
- Compare average popularity across genres.
- Analyze relationships between audio features and popularity.
- Investigate correlations between numerical audio characteristics.
- Apply clustering to identify groups of tracks with similar audio properties.
- Evaluate cluster separation using Silhouette Scores.
- Visualize clustering results using Principal Component Analysis (PCA).

## Dataset

**Source:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

**Creator:** Maharshi Pandya

The original dataset includes approximately 114,000 records and 21 columns.

Selected variables include:

| Feature | Description |
|---|---|
| `track_name` | Track title |
| `artists` | Performing artist(s) |
| `track_genre` | Music genre |
| `popularity` | Spotify popularity score |
| `duration_ms` | Track duration in milliseconds |
| `danceability` | Danceability score |
| `energy` | Perceived intensity and activity |
| `acousticness` | Acoustic confidence score |
| `instrumentalness` | Instrumental content estimate |
| `valence` | Musical positivity |
| `tempo` | Estimated beats per minute |

A track may appear in multiple genre categories. Therefore, record counts do not necessarily represent unique songs.

## Methodology

### 1. Data Cleaning

The data cleaning pipeline performs:

- Missing value inspection and handling.
- Removal of unnecessary index columns.
- Removal of exact duplicate records.
- Validation of numerical values.
- Validation of audio feature ranges.
- Conversion of track duration into minutes.
- Export of a cleaned CSV dataset.

**Script:** `src/data_cleaning.py`

### 2. Exploratory Data Analysis

The EDA pipeline generates visualizations covering:

- Top 10 music genres by record count.
- Distribution of popularity scores.
- Top 10 genres by average popularity.
- Energy versus popularity.
- Danceability versus popularity.
- Pearson correlation matrix of selected audio features.

It also exports summary statistics as CSV files.

**Script:** `src/eda.py`

### 3. Machine Learning — Music Clustering

Unsupervised learning is implemented using MiniBatch K-Means.

The workflow includes:

1. Selection of seven numerical audio features.
2. Feature standardization using StandardScaler.
3. Evaluation of cluster counts from 2 to 8 on a sample.
4. Selection based on the Silhouette Score.
5. Clustering of the cleaned dataset.
6. Generation of cluster-level feature summaries.
7. PCA visualization in two dimensions.

The resulting clusters represent numerical similarities in audio features, rather than predefined musical genres.

**Script:** `src/clustering.py`

## Technologies

| Category | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Clustering | MiniBatch K-Means |
| Dimensionality Reduction | PCA |
| Cluster Evaluation | Silhouette Score |
| Version Control | Git and GitHub |

## Project Structure

```text
spotify-music-data-analysis/
├── data/
│   ├── dataset.csv
│   └── README.md
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   └── clustering.py
├── .gitignore
├── requirements.txt
└── README.md
```

The analysis scripts generate a cleaned dataset and a `results/` directory containing visualizations and summary statistics.

## Installation

Clone the repository:

```bash
git clone https://github.com/Nikol24/spotify-music-data-analysis.git
```

Navigate to the project directory:

```bash
cd spotify-music-data-analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Analysis

Execute the following scripts in order:

**Step 1 — Data Cleaning**

```bash
python src/data_cleaning.py
```

**Step 2 — Exploratory Data Analysis**

```bash
python src/eda.py
```

**Step 3 — Machine Learning Clustering**

```bash
python src/clustering.py
```

The data cleaning script must run first because the EDA and clustering scripts depend on its output.

## Generated Outputs

The EDA pipeline generates:

- `top_10_genres.png`
- `popularity_distribution.png`
- `genre_popularity.png`
- `energy_vs_popularity.png`
- `danceability_vs_popularity.png`
- `correlation_matrix.png`
- `top_10_genres.csv`
- `genre_popularity.csv`
- `correlation_matrix.csv`

The clustering pipeline generates:

- `spotify_clusters_pca.png`
- `cluster_summary.csv`
- `silhouette_scores.csv`

These files are generated when the scripts are executed successfully.

## Limitations

- Spotify popularity scores can change over time.
- Repeated track identifiers may occur across different genre categories.
- Genre-level record counts should not be interpreted as counts of unique songs.
- Correlation does not imply causation.
- Cluster assignments depend on the selected features, scaling method, and clustering algorithm.
- Silhouette Scores describe cluster separation but do not guarantee meaningful musical categories.
- PCA visualizations are two-dimensional approximations of a higher-dimensional feature space.

## Author

**GitHub:** [Nikol24](https://github.com/Nikol24)

*Data Science Portfolio — Spotify Music Analysis & Unsupervised Machine Learning*
