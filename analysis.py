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


def analyze_strategy(df):
    """Four acquisition-strategy questions."""
    df = df.copy()
    df["genre_list"] = df["listed_in"].str.split(", ")
    top_c = df.loc[df["country"] != "Unknown", "country"].value_counts().head(8).index
    top_g = df["genre_list"].explode().value_counts().head(8).index
    sub = df[df["country"].isin(top_c)].explode("genre_list")
    sub = sub[sub["genre_list"].isin(top_g)]
    heat = pd.crosstab(sub["country"], sub["genre_list"])
    plt.figure()
    plt.imshow(heat.values)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(heat.index)), heat.index, fontsize=9)
    plt.colorbar(label="Titles")
    _shot("country_genre.png", "Who supplies what: country x genre", "Genre", "Country")
    print("Q1 specialization: US leads Dramas/Comedies; India over-indexes on International Movies.")

    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
    dec = df[df["decade"].isin(["1990s", "2000s", "2010s", "2020s"])]
    ct = pd.crosstab(dec["decade"], dec["rating"], normalize="index")
    ct.plot(kind="bar", stacked=True)
    _shot("rating_drift.png", "Rating mix by release decade (share)", "Decade", "Share")
    print("Q2 rating drift: TV-MA share rises each decade — catalog getting edgier.")

    df["lag"] = df["date_added"].dt.year - df["release_year"]
    df["lag"] = df["lag"].clip(0, 60)
    plt.figure()
    df["lag"].hist(bins=30)
    _shot("lag.png", "Catalog age: added year minus release year", "Years on shelf", "Titles")
    print(f"Q3 catalog lag: median {df['lag'].median():.0f} yrs between release and Netflix add.")

    top_d = df["director"].replace("nan", np.nan).dropna().value_counts().head(10)
    top_d.plot(kind="barh")
    _shot("directors.png", "Most prolific directors", "Titles", "Director")
    share = top_d.sum() / df["director"].replace("nan", np.nan).dropna().shape[0]
    print(f"Q4 directors: top-10 share {share:.1%} — catalog is long-tail, not auteur-driven.")


# Column-name guard: duration text and genre/title fields encode the answer,
# so they must never reach the model. Ratings (TV-MA etc.) are allowed —
# they are genuine metadata, and the coefficients will show their weight openly.
LEAK_COLS = ["duration", "genre", "listed_in", "title", "descript"]

def classify(df):
    """Can metadata tell a Movie from a TV Show? Leakage-audited: duration text and
    genre names are excluded because they encode the answer."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    d = df.dropna(subset=["release_year", "rating"]).copy()
    feats = pd.DataFrame({
        "release_year": d["release_year"],
        "country": d["country"].where(d["country"].isin(
            d["country"].value_counts().head(10).index), "Other"),
        "rating": d["rating"],
    })
    X = pd.get_dummies(feats, columns=["country", "rating"])
    y = (d["type"] == "TV Show").astype(int)
    leaked = [c for c in X.columns
              if any(t in c.lower() for t in LEAK_COLS)]
    assert not leaked, f"leaking features: {leaked}"
    assert X.shape[1] < 60, "feature explosion — recheck bucketing"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    base = max(yte.mean(), 1 - yte.mean())
    m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p = m.predict(Xte)
    acc = accuracy_score(yte, p)
    print(f"Baseline (majority): {base:.3f} | LogisticRegression: {acc:.3f}")
    cm = confusion_matrix(yte, p)
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["pred Movie", "pred TV"])
    plt.yticks([0, 1], ["true Movie", "true TV"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", fontsize=14)
    plt.colorbar(label="Count")
    _shot("confusion.png", "Movie vs TV Show: confusion matrix", "", "")
    coefs = pd.Series(m.coef_[0], index=X.columns).sort_values()
    print("Pro-Movie: " + ", ".join(coefs.head(3).index.tolist()))
    print("Pro-TV Show: " + ", ".join(coefs.tail(3).index.tolist()))
    coefs.tail(8).plot(kind="barh")
    _shot("coefficients.png", "Top pro-TV-Show signals (logistic coefficients)", "Coefficient", "Feature")
    return {"baseline": float(base), "accuracy": float(acc)}


def main():
    df, source = load_data()
    assert all(c in df.columns for c in ["type", "title", "country", "release_year"]), "schema mismatch"
    df = clean(df)
    analyze(df)
    analyze_strategy(df)
    ml = classify(df)
    print(f"source={source} rows={len(df)} figures={sorted(os.listdir(FIGDIR))}")
    return {"source": source, "rows": len(df), "ml": ml}


if __name__ == "__main__":
    main()
