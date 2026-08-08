# Customer Segmentation usinh K-Means

Advanced customer segmentation using Behavioural Features + K-Means++ + SHAP.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-lightgrey.svg)](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## Problem Statement

Retailers need to group customers into actionable segments to target marketing spend, retention campaigns, and inventory planning. The standard approach **RFM (Recency, Frequency, Monetary)** is simple and widely used, but it has real limitations:

- **Blind to behaviour, not just value.** Two customers with identical RFM scores can behave very differently: one buys the same items repeatedly, the other explores and returns most of what they order. RFM cannot tell them apart.
- **No signal on returns.** Return rate is often one of the strongest predictors of true customer value and churn risk, yet it is absent from the RFM feature set entirely.
- **One-size-fits-all clustering.** Applying a single clustering pass to the whole customer base mixes retail customers with wholesale-like/reseller accounts that have fundamentally different purchase patterns (bulk orders, high SKU concentration), distorting the segments for everyone else.
- **Segments aren't explainable.** Even when clustering succeeds, stakeholders need to know *why* a customer landed in a given segment; a plain cluster label isn't enough to justify a business action.

**Goal:** build a segmentation pipeline that (1) separates structurally different customer types before clustering, (2) uses a richer, purpose-built feature set beyond RFM, and (3) explains what actually drives each segment.

---

## Approach

**1. Behavioural feature engineering (20 → 14 features).**
Beyond RFM, engineer features across purchase rhythm, spending shape, basket behaviour, volume/bulk buying, product concentration, customer lifecycle, and spend acceleration — then drop redundant/correlated features via a correlation filter, keeping a compact set of 14.

**2. Two-stage clustering.**
- **Stage 1 (rule-based):** split off wholesale-like accounts using SKU concentration (`SKU_HHI > 0.5`) before any clustering happens, so bulk buyers don't distort the retail segmentation.
- **Stage 2 (K-Means++, k=4):** cluster the remaining retail customers on the 14 behavioural features to produce four interpretable segments.

**3. Explainability with SHAP.**
Train a Random Forest as a surrogate classifier over the cluster assignments, then use SHAP (`TreeExplainer`) to quantify which features actually drive each segment — turning "customer X is in cluster 2" into "customer X is in cluster 2 mainly because of high `ReturnRate`."

**4. Business profiling.**
Translate clusters back into raw, business-readable metrics (not just scaled features) and pair each segment with a concrete strategy recommendation.

---

## Contents

| Part | Topic | Notebook |
|------|-------|----------|
| 2 | EDA & Data Cleaning | [`01_cleaning_and_eda.ipynb`](notebooks/01_cleaning_and_eda.ipynb) |
| 3 | Feature Engineering (20 → 14 features) | [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) |
| 4 | Two-Stage Clustering (K-Means++) | [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) |
| 5 | SHAP Explainability — Opening the black box | [`04_shap_analysis.ipynb`](notebooks/04_shap_analysis.ipynb) |
| 6 | Cluster Profiles & Strategy (raw-value) | [`05_cluster_profiling.ipynb`](notebooks/05_cluster_profiling.ipynb) |

---

## Dataset

Download the **UCI Online Retail** dataset and place it at `data/raw/online_retail.csv`:

> [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset itself is not stored in the repo (see `.gitignore`).

| Info | Value |
|------|-------|
| Retailer | UK-based online, gifts & housewares |
| Period | Dec 2010 – Dec 2011 |
| Raw rows | 541,909 transactions |
| After cleaning | 354,321 transactions, 3,920 UK customers |

---

## Environment Setup

```bash
# Using uv (recommended)
uv sync --dev

# Or pip
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

---

## Running the Notebooks

Run in order — each notebook saves output consumed by the next:

```
01_cleaning_and_eda.ipynb      →  data/processed/cleaned_uk_data.csv
02_feature_engineering.ipynb   →  data/processed/customer_features_*_scaled.csv
03_modeling.ipynb              →  data/processed/cluster_assignments_retail_k4.csv
04_shap_analysis.ipynb         →  figures/04_shap/
05_cluster_profiling.ipynb     →  data/processed/cluster_customer_profiles.csv + cluster_strategy_summary.csv
```

---

## Key Results

**Two-stage pipeline:**
1. **Stage 1 (rule-based):** split off 179 wholesale-like accounts (`SKU_HHI > 0.5`) — 4.6% of the population
2. **Stage 2 (K-Means++, k=4):** clustered 3,741 retail customers on 14 behavioural features

**4 retail segments:**

| Cluster | Name | n (%) | Key signal |
|---------|------|-------|------------|
| C0 | Seasonal Intensives | 330 (8.8%) | `MonthlyOrderRate` ↑, `QuarterConcentration` ↑ (Q4-heavy buying) |
| C1 | High-Return Actives | 1,107 (29.6%) | `ReturnRate` ↑ — **absent from RFM**; highest LTV |
| C2 | One-Time Explorers | 1,257 (33.6%) | `BasketSizeCV` ≈ 0, `Recency` ↑, no repeat SKUs (one-time buyers) |
| C3 | Occasional Loyalists | 1,047 (28.0%) | Zero returns, long tenure, moderate frequency |

**SHAP surrogate:** RF with 200 trees, 99.0% OOF accuracy — `ReturnRate` accounts for ~27% of global feature importance.

---

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

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` ≥2.2, `numpy` ≥2.0 | Data processing |
| `scikit-learn` ≥1.5 | K-Means++, Random Forest, scalers, metrics |
| `shap` ≥0.46 | SHAP TreeExplainer |
| `matplotlib` ≥3.9, `seaborn` ≥0.13 | Static visualisation |
| `plotly` ≥5.23 | Interactive visualisation |
| `scipy` ≥1.13 | Statistical utilities |

---

## License

MIT — see [LICENSE](LICENSE).
