# Netflix Movies & Shows Explorer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harrisonokojiee/netflix-titles/blob/main/netflix_explorer.ipynb)

Portfolio project 2: what is Netflix actually buying? Content mix, additions over
time, top countries, genres, ratings, and durations — an acquisition-strategy read
of the catalog.

## Business questions
1. Movies vs TV Shows: where is the catalog weight?
2. Which countries and genres dominate acquisitions?
3. How has the additions pace moved year to year?
4. What maturity ratings and runtimes define the catalog?

## Dataset
- Kaggle: `shivamb/netflix-shows` — Netflix Titles (~8,800 rows x 12 cols).
- The notebook fetches it with `kagglehub.dataset_download(...)`. Without Kaggle
  auth it falls back to a synthetic same-schema sample so Run All always works.

## Run in Colab
1. Open `netflix_explorer.ipynb` in Colab and Run All.
2. For real data: add your Kaggle token via Colab Secrets (`KAGGLE_USERNAME` /
   `KAGGLE_KEY`) or upload `kaggle.json` when prompted.

## Run locally
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```
Figures land in `figures/`.

## Key findings (synthetic fallback run; real data decides)
- ~70/30 movie-to-show split; median movie 125 min, median show 3 seasons.
- US and India lead producing countries; Comedies and Documentaries top genres.
- Family-heavy ratings mix (PG-13/PG/TV-PG/G all large) — breadth strategy, not edgy exclusives.

## Skills shown
pandas cleaning, categorical EDA, matplotlib storytelling, Kaggle ingest, Colab reproducibility. Companion to the [King County housing project](../house-price-analysis/).
