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

## Key findings (real run: 8,807 titles)
- 6,131 Movies vs 2,676 TV Shows (~70/30); median movie 98 min, median show 1 season.
- US (2,818) and India (972) lead supply, then UK (419) and Japan (245); International Movies (2,752) top genre ahead of Dramas (2,427).
- Adult-skewed ratings (TV-MA 3,207, TV-14 2,160) — edgier than a family catalog; additions peaked 2019 (1,999) then eased through 2021.
- Specialization: the US feeds Dramas/Comedies while India over-indexes on International Movies; ratings get edgier by decade.
- Catalog is fresh (median 1 year from release to Netflix) and long-tail (top-10 directors hold just 2.3%).

## Modeling: can metadata tell a Movie from a TV Show?
- LogisticRegression on release_year + country + rating: **0.702** vs 0.679 majority baseline — metadata barely separates format, stated plainly.
- Strongest signals: R/PG-13/PG ratings → Movie; South Korea, TV-Y/TV-Y7 → TV Show.
- Leakage audit enforced in code: duration text and genre/title fields are excluded because they encode the answer (genre names literally contain "Movies"/"TV Shows"); ratings stay as genuine metadata.

## Conclusion
Netflix buys breadth, not auteurs: a fresh, long-tail catalog led by US drama/comedy and Indian international titles, trending edgier by decade. Format is barely predictable from metadata alone (70.2% vs 67.9% baseline) — the catalog's variety is the finding. All numbers from the real Kaggle data; rerun via Colab or `python analysis.py`.

## Skills shown
pandas cleaning, categorical EDA, matplotlib storytelling, Kaggle ingest, Colab reproducibility. Companion to the [King County housing project](../house-price-analysis/).
