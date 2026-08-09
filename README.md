# Customer Segmentation using K-Means

Advanced customer segmentation using Behavioural Features + K-Means++ + SHAP.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-lightgrey.svg)](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## Problem Statement

An online retailer (~4,000 customers, ~500,000 transactions) can't target marketing effectively: the same message goes to everyone, so one-time buyers get "loyal customer" perks, wholesale accounts get single-item promos, and Q4-only shoppers get contacted in July.

The standard fix, **RFM (Recency, Frequency, Monetary)**, doesn't hold up:
- **Penalises returns unfairly** - cancellations reduce Frequency & Monetary, mislabeling active explorers as low-value.
- **Misreads seasonality** - Q4-only buyers look like they're churning the rest of the year.
- **Can't separate wholesale from retail** - a reseller and an ordinary shopper can land on the same score.
- **No behavioural signal** - only totals; no sense of rhythm, basket variety, or spend trend.
- **Not explainable** - a segment label with no reason attached isn't actionable.

**Goal:** segment by actual behaviour, separate structurally different customer types before clustering, and explain what drives each segment well enough to act on.

---

## Approach

| Step | Question | Output |
|------|----------|--------|
| 1. EDA & Cleaning | Is the raw data trustworthy? | Cleaned UK-only transactions |
| 2. Feature Engineering | What does each customer's behaviour look like? | 20 candidate features → 14 after correlation filtering |
| 3. K-Means Clustering | Which customers behave alike? | 4 retail segments (after rule-based wholesale split) |
| 4. SHAP Explainability | *Why* did a customer land there? | Per-feature, per-cluster attribution |
| 5. Strategy | What should marketing do about it? | One action per segment |

## Notebook Contents

| Part | Topic | Notebook |
|------|-------|----------|
| 2 | EDA & Data Cleaning | [`01_cleaning_and_eda.ipynb`](notebooks/01_cleaning_and_eda.ipynb) |
| 3 | Feature Engineering | [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) |
| 4 | Two-Stage Clustering (K-Means++) | [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) |
| 5 | SHAP Explainability | [`04_shap_analysis.ipynb`](notebooks/04_shap_analysis.ipynb) |
| 6 | Cluster Profiles & Strategy | [`05_cluster_profiling.ipynb`](notebooks/05_cluster_profiling.ipynb) |

---

## Dataset

**[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)** - UK gift/houseware retailer, Dec 2010–Dec 2011. Not stored in the repo; download and place at `data/raw/online_retail.csv` (see `.gitignore`).

541,909 raw transactions → **354,321 transactions, 3,920 UK customers** after cleaning: drop cancelled invoices (`InvoiceNo` starts with `C`), keep UK only, drop missing `CustomerID` (~25%), drop `Quantity`/`UnitPrice` ≤ 0. One-time buyers are kept - they're a real segment, not noise.

---

## Feature Engineering

Design principle: not *"what does this measure?"* but *"what does this tell us about behaviour?"* 20 candidate features across 5 groups, filtered down to **14** after removing pairs with `|r| > 0.85`:

| Group | Question | Features |
|-------|----------|----------|
| Purchase Rhythm | How often/predictably do they buy? | `Recency`, `InterPurchaseCV`, `ActiveSpanDays`, `MonthlyOrderRate` |
| Spending Shape | How is money spent? | `AOV`, `SpendGini`, `PriceCV`, `MaxSpendShare`, `SpendAcceleration` |
| Basket Behaviour | What do they buy each visit? | `NewSKURate`, `RepeatSKUFraction`, `BasketSizeCV`, `MedianBasketQty` |
| Volume & Bulk (B2B) | Is this a wholesale account? | `BurstIndex`, `BulkLineRate`, `QuantityCV`, `SKU_HHI` |
| Customer Lifecycle | What's invisible to RFM? | `ReturnRate`, `ReturnValueRate`, `QuarterConcentration` |

Two features worth calling out: **`SKU_HHI`** (Herfindahl-Hirschman Index borrowed from antitrust economics) measures product concentration - above 0.5 is a strong B2B signal. **`ReturnRate`** is the single most important feature in the final model.

Preprocessing order matters: `QuantileTransformer` (fixes skew/zero-inflation) → `StandardScaler` (required for K-Means' Euclidean distance) → clip to ±3σ (bounds extreme B2B outliers without dropping them).

---

## Clustering Methodology

**K-Means** was chosen over DBSCAN/Hierarchical/Gaussian Mixture for simplicity and centroids that are directly interpretable by business stakeholders.

**Wholesale accounts are split off by rule, not ML** - the signal (`SKU_HHI > 0.5`) is clear-cut, domain knowledge is available, and the class is too small a minority (4.6%) for a general model to learn reliably. Leaving them in hurts clustering quality: Silhouette 0.177 mixed vs. **0.244** on the 3,741 retail-only customers once removed.

**k=4** was chosen over the statistically-optimal k=2 (Silhouette 0.315, but only separates "buys a lot" from "buys a little"). Going from k=3 to k=4 splits out a distinct, actionable Q4-seasonal segment - balancing statistical quality against business usefulness rather than optimising the metric alone.

---

## Key Results

179 wholesale-like accounts (`SKU_HHI > 0.5`) split off first; the remaining 3,741 retail customers cluster into 4 segments:

| Cluster | Name | n (%) | Key signal | Strategy |
|---------|------|-------|------------|----------|
| C0 | Seasonal Intensives | 330 (8.8%) | `QuarterConcentration` ↑ (Q4-heavy) | Pre-season campaign + year-round loyalty; don't judge by Recency outside Q4 |
| C1 | High-Return Actives | 1,107 (29.6%) | `ReturnRate` ↑ - absent from RFM | Better product info, no return-fee penalties - they're still exploring |
| C2 | One-Time Explorers | 1,257 (33.6%) | No repeat SKUs, short tenure | Category-based re-engagement, not generic discounts |
| C3 | Occasional Loyalists | 1,047 (28.0%) | Zero returns, long tenure, highest LTV | Light touch - exclusive access, not discounts |

---

## SHAP Explainability

K-Means only returns a cluster label - no reason attached. A Random Forest surrogate (200 trees, 0.9904 ± 0.0023 CV accuracy) reproduces the cluster assignments closely enough that SHAP on it explains the clustering itself.

**Global importance:** `ReturnRate` 27.5%, `QuarterConcentration` 21.0%, `BasketSizeCV` 18.4%, `RepeatSKUFraction` 11.2%, `MonthlyOrderRate` 10.0% - these top 3 alone account for roughly two-thirds of total discriminating power. Per cluster: C1 is driven by high `ReturnRate`, C0 by high `QuarterConcentration`, C2 by low `BasketSizeCV`/short tenure, C3 by the absence of the other three signals.

---

## Environment Setup

```bash
uv sync --dev          # recommended
pip install -r requirements.txt   # or pip
```

Requires Python ≥ 3.10.

## Running the Notebooks

Run in order - each saves output the next one consumes:

```
01_cleaning_and_eda.ipynb      →  data/processed/cleaned_uk_data.csv
02_feature_engineering.ipynb   →  data/processed/customer_features_*_scaled.csv
03_modeling.ipynb              →  data/processed/cluster_assignments_retail_k4.csv
04_shap_analysis.ipynb         →  figures/04_shap/
05_cluster_profiling.ipynb     →  data/processed/cluster_customer_profiles.csv + cluster_strategy_summary.csv
```

## Project Structure

```
advanced_customer_segmentation/
├── notebooks/              # 5 lecture notebooks (01 → 05)
├── src/
│   ├── clustering_library/ # DataCleaner, FeatureEngineer, ClusterAnalyzer, DataVisualizer
│   ├── notebook_io.py      # Path read/write utilities
│   └── visual_style.py     # Shared colours and plotting style
├── data/
│   ├── raw/                # online_retail.csv (manual download)
│   └── processed/          # Generated by running the notebooks
├── figures/                # Generated by running the notebooks
├── pyproject.toml
└── requirements.txt
```

## Tech Stack

`pandas` ≥2.2 · `numpy` ≥2.0 · `scikit-learn` ≥1.5 · `shap` ≥0.46 · `matplotlib` ≥3.9 · `seaborn` ≥0.13 · `plotly` ≥5.23 · `scipy` ≥1.13

## License

MIT - see [LICENSE](LICENSE).
