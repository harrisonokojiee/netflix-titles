"""Netflix Movies & Shows Explorer — content strategy EDA.
Runs locally (.venv) and in Colab. Fetches Kaggle data when credentials
exist, otherwise uses a synthetic fallback with the same schema.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11})
SOURCE_TAG = "Source: Kaggle shivamb/netflix-shows or same-schema fallback"

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
DATADIR = os.path.join(HERE, "data")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)


def _shot(name, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.figtext(0.01, 0.01, SOURCE_TAG, fontsize=8, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close()


def load_data(n_fallback=6000, seed=7):
    for f in os.listdir(DATADIR) if os.path.isdir(DATADIR) else []:
        if f.endswith(".csv"):
            p = os.path.join(DATADIR, f)
            df = pd.read_csv(p)
            print(f"Loaded local CSV: {p} {df.shape}")
            return df, "local_csv"
    try:
        import kagglehub
        path = kagglehub.dataset_download("shivamb/netflix-shows")
        csvs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        if csvs:
            df = pd.read_csv(csvs[0])
            print(f"Loaded Kaggle: {csvs[0]} {df.shape}")
            return df, "kagglehub"
    except Exception as e:
        print(f"kagglehub fetch skipped ({type(e).__name__}: {e})")
    rng = np.random.default_rng(seed)
    n = n_fallback
    types = rng.choice(["Movie", "TV Show"], size=n, p=[0.7, 0.3])
    genres = ["Dramas", "Comedies", "Documentaries", "Action & Adventure",
              "Horror Movies", "Romantic Movies", "Sci-Fi & Fantasy",
              "Crime TV Shows", "Kids' TV", "Stand-Up Comedy"]
    countries = ["United States", "India", "United Kingdom", "Canada", "France",
                 "Japan", "Spain", "South Korea", "Germany", "Mexico"]
    ratings = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "PG", "TV-Y7", "G"]
    df = pd.DataFrame({
        "show_id": [f"s{i}" for i in range(1, n + 1)],
        "type": types,
        "title": [f"Title {i}" for i in range(1, n + 1)],
        "director": rng.choice(["Dir A", "Dir B", "Dir C", "Dir D", np.nan], size=n),
        "cast": rng.choice(["Cast X", "Cast Y", "Cast Z", np.nan], size=n),
        "country": rng.choice(countries + [np.nan], size=n,
                              p=[0.14, 0.12, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05, 0.21]),
        "date_added": pd.to_datetime(rng.choice(pd.date_range("2015-01-01", "2021-12-31"), n)),
        "release_year": rng.integers(1990, 2022, n),
        "rating": rng.choice(ratings, size=n),
        "duration": [f"{m} min" if t == "Movie" else f"{s} Season{'s' if s > 1 else ''}"
                     for t, m, s in zip(types, rng.integers(70, 180, n), rng.integers(1, 6, n))],
        "listed_in": [", ".join(rng.choice(genres, size=k, replace=False))
                      for k in rng.integers(1, 4, n)],
        "description": ["A story of " + w for w in rng.choice(
            ["love", "war", "family", "crime", "space", "music"], size=n)],
    })
    print(f"Using synthetic fallback {df.shape} (real set is ~8800 rows x 12 cols)")
    return df, "synthetic"


def clean(df):
    before = len(df)
    df = df.copy()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["title", "type"])
    df["country"] = df["country"].replace(["nan", "None", ""], np.nan).fillna("Unknown")
    df["release_year"] = df["release_year"].clip(1925, 2021)
    df = df.drop_duplicates(subset=["show_id"])
    print(f"Cleaning: {before} -> {len(df)} rows")
    return df


def analyze(df):
    print("Content mix:\n" + df["type"].value_counts().to_string())
    df["added_year"] = df["date_added"].dt.year
    print("Additions per year:\n" + df["added_year"].value_counts().sort_index().to_string())
    print("Top countries:\n" + df.loc[df["country"] != "Unknown", "country"].value_counts().head(8).to_string())
    print("Top ratings:\n" + df["rating"].value_counts().head(6).to_string())
    movies = df[df["type"] == "Movie"].copy()
    movies["mins"] = movies["duration"].str.extract(r"(\d+)").astype(float)
    print(f"Median movie length: {movies['mins'].median():.0f} min")
    shows = df[df["type"] == "TV Show"].copy()
    shows["seasons"] = shows["duration"].str.extract(r"(\d+)").astype(float)
    print(f"Median TV seasons: {shows['seasons'].median():.0f}")
    print("Top genres:\n" + df["listed_in"].str.split(", ").explode().value_counts().head(8).to_string())

    df["type"].value_counts().plot(kind="bar")
    _shot("mix.png", "Movies vs TV Shows", "Type", "Titles")

    df["added_year"].value_counts().sort_index().plot(marker="o")
    _shot("additions.png", "Titles added per year", "Year added", "Titles")

    df.loc[df["country"] != "Unknown", "country"].value_counts().head(10).plot(kind="barh")
    _shot("countries.png", "Top producing countries", "Titles", "Country")

    df["listed_in"].str.split(", ").explode().value_counts().head(10).plot(kind="barh")
    _shot("genres.png", "Top genres", "Titles", "Genre")

    df["rating"].value_counts().head(8).plot(kind="bar")
    _shot("ratings.png", "Content by maturity rating", "Rating", "Titles")

    plt.figure()
    movies["mins"].hist(bins=30)
    _shot("durations.png", "Movie length distribution", "Minutes", "Movies")


def main():
    df, source = load_data()
    assert all(c in df.columns for c in ["type", "title", "country", "release_year"]), "schema mismatch"
    df = clean(df)
    analyze(df)
    print(f"source={source} rows={len(df)} figures={sorted(os.listdir(FIGDIR))}")
    return {"source": source, "rows": len(df)}


if __name__ == "__main__":
    main()
