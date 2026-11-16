# Netflix User Rating Analytics & SVD Movie Recommendation Engine

An end-to-end exploratory data analysis (EDA) and collaborative filtering recommendation system built on **662,313 Netflix user ratings across 244,733 customers and 175 movies**, utilizing quantile-based sparsity filtering and **TruncatedSVD Matrix Factorization**.

## System Architecture Flow

```text
Raw Netflix Prize Logs (combined_data_1.txt: 662,488 rows)
        │
        ▼
Header Extraction & NaN Parsing (175 Movie IDs, 244,733 Users, 662,313 Ratings)
        │
        ▼
Quantile Sparsity Filtering (Top 30% Active Movies >= 1,241 & Users >= 3 ratings)
        │
        ▼
User-Item Interaction Pivot Matrix (86,998 Users x 53 Movies)
        │
        ├───────────────────────────────────────────┐
        ▼                                           ▼
Popularity & Satisfaction Ranking          TruncatedSVD Matrix Factorization
(Most Rated vs. Highest/Lowest Mean)       (Latent Factor Rating Reconstruction)
                                                    │
                                                    ▼
                                      Top-N Personalized Movie Recommendations
```

## Key Findings & Metrics
- **Rating Distribution (`662,313` total ratings):**
  - `4.0 Stars`: `232,420` (~35%)
  - `3.0 Stars`: `184,810` (~28%)
  - `5.0 Stars`: `149,115` (~23%)
  - `2.0 Stars`: `64,654` (~10%)
  - `1.0 Star`: `31,314` (~5%)
- **Sparsity Reduction:** Applied 70th-percentile thresholds (`>= 1,241` reviews per movie; `>= 3` reviews per customer) to distill `662,313` raw interactions into a dense `86,998 × 53` user-item rating matrix (`416,461` active ratings).
- **Top Popular vs. Highest-Rated Titles:**
  - *Most Rated:* **Something's Gotta Give** (`57,682` ratings), **Reservoir Dogs** (`45,751` ratings), **The Game** (`30,477` ratings).
  - *Highest Satisfaction (>500 reviews):* **Invader Zim** (`4.11 / 5.0`), **Aqua Teen Hunger Force: Vol. 1** (`4.11 / 5.0`).

## Tech Stack
- **Data Wrangling & Matrix Operations:** Python, Pandas, NumPy
- **Collaborative Filtering:** Scikit-Learn (`TruncatedSVD`)
- **Visualization:** Matplotlib
