# Customer Segmentation using K-Means

Advanced customer segmentation using Behavioural Features + K-Means++ + SHAP.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-lightgrey.svg)](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## Problem Statement

**Business context.** An online retailer has ~4,000 customers and ~500,000 transactions, but no way to segment them. Marketing cost keeps rising while spend becomes less effective - the same message goes out to everyone.

**What one-size-fits-all marketing looks like in practice:**
- One email blast to the entire customer base → poor conversion.
- One-time buyers still receive "loyal customer" perks → wasted budget, wrong offer.
- Wholesale buyers still receive single-item promotions → irrelevant, drives unsubscribes.
- Customers who only buy in Q4 get contacted in July → wrong timing, no effect.
- Customers with many returns get labelled "bad" and dropped → a potentially high-value segment gets missed.

**Why the standard fix - RFM (Recency, Frequency, Monetary) - falls short.** RFM buckets each of the three dimensions into 5 quintiles and combines them into a 3-digit score. It's popular because it only needs basic transaction data (date, order count, revenue), which made it the industry default since the 1990s. But it has real limitations:

- **Unfairly penalises high-return customers.** Cancelled/returned orders reduce both Frequency and Monetary, so RFM mislabels customers who are actively exploring the catalogue as low-value.
- **Misreads seasonal buyers.** Customers who only purchase in a fixed season (e.g. Q4) rack up high Recency between orders and get flagged as "about to churn" by RFM, even though that's simply their normal purchase cycle.
- **Cannot separate wholesale from retail customers.** Two customers with fundamentally different purchasing behaviour - a reseller placing bulk orders and an ordinary shopper - can land on identical RFM scores.
- **Only captures totals, not behavioural trends.** RFM has no way to measure whether purchases arrive at a steady rhythm or in bursts, how varied a customer's basket is, or whether their spending is trending up or down.
- **Segments aren't explainable.** Even when clustering succeeds, stakeholders need to know *why* a customer landed in a given segment - a plain cluster label isn't enough to justify a business action.

**Goal:** replace "one message for everyone" with "the right message for the right group" - segment customers by actual behaviour (not just RFM totals), separate structurally different customer types before clustering, and explain what drives each segment well enough for a marketing team to act on it.

---

## Approach

Five steps, each answering a specific question:

| Step | Question it answers | Output |
|------|---------------------|--------|
| 1. EDA & Cleaning | Is the raw data trustworthy? | Cleaned UK-only transaction table |
| 2. Feature Engineering | What does each customer's behaviour actually look like? | 20 candidate behavioural features → 14 after correlation filtering |
| 3. K-Means Clustering | Which customers behave alike? | 4 interpretable retail segments (after rule-based wholesale split) |
| 4. SHAP Explainability | *Why* did a customer land in that segment? | Per-feature, per-cluster attribution |
| 5. Strategy | What should marketing actually do for each segment? | One concrete action per segment |

---

## Notebook Contents

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
| Period | Dec 2010 – Dec 2011 (13 months) |
| Raw rows | 541,909 transactions |
| After cleaning | 354,321 transactions, 3,920 UK customers |

| Column | Meaning | Example | Note |
|--------|---------|---------|------|
| `InvoiceNo` | Invoice ID | `536365` / `C536379` | Starts with `C` → cancelled |
| `CustomerID` | Customer ID | `17850` | ~25% missing → dropped |
| `Quantity` | Units purchased | `6` | Negative → stock adjustment, not a real sale |
| `UnitPrice` | Unit price (£) | `2.55` | `≤ 0` → invalid/test entry, dropped |
| `Country` | Country | `United Kingdom` | Only UK is kept for modelling |
| `InvoiceDate` | Timestamp | `01/12/2010 08:26` | Dec 2010 – Dec 2011 |

### Data cleaning funnel

| Step | Rule | Rows before → after | Removed |
|------|------|----------------------|---------|
| 1 | Drop cancelled invoices (`InvoiceNo` starts with `C`) | 541,909 → 532,621 | 9,288 (1.7%) |
| 2 | Keep `Country == "United Kingdom"` only | 532,621 → 487,622 | 44,999 (8.3%) |
| 3 | Drop missing `CustomerID` | 487,622 → 354,345 | 133,277 (24.6%) |
| 4 | Drop `Quantity ≤ 0` and `UnitPrice ≤ 0` | 354,345 → 354,321 | 24 |
| 5 | Keep customers with a single purchase | — | none — one-time buyers are a real segment, not noise |

Rationale: invoices starting with `C` are cancellations (`Quantity` is negative and doesn't reflect a real sale); non-UK customers behave differently (recurring wholesale) and are excluded to keep the model focused on the primary market; rows without a `CustomerID` can't be tied to a customer's behaviour over time; `Quantity`/`UnitPrice ≤ 0` rows are internal stock adjustments or test entries, not purchases.

---

## Exploratory Findings

Three patterns discovered during EDA directly shaped the feature engineering and preprocessing decisions below:

- **Strong Q4 seasonality.** Monthly revenue climbs sharply from September, peaks in November (pre-Christmas shopping), and troughs in January–February. This is typical of a gift retailer, and it motivated `QuarterConcentration` - a feature measuring how much of a customer's annual spend lands in a single quarter.
- **Weekday, office-hours purchase pattern.** Transaction volume is concentrated 9am–5pm, Monday–Friday, with almost none on weekends or evenings — a signature of B2B/reseller ordering rather than personal (B2C) shopping. This motivated the "is this a B2B customer?" feature group (`SKU_HHI`, `BulkLineRate`, `QuantityCV`, `BurstIndex`) and the decision to split wholesale-like accounts out before clustering.
- **Heavy-tailed spending.** Median customer spend is ~£300, but the top 1% spend over £10,000 — 30x the median. Since K-Means relies on Euclidean distance, this kind of outlier would pull cluster centroids off-target, which motivated the `QuantileTransformer` step in preprocessing.

A standard RFM segmentation map (10 segments - Champions, Loyal, At Risk, Hibernating, etc.) was also built as a baseline, to check whether the richer behavioural feature set actually improves on it.

---

## Feature Engineering

**Design principle:** don't ask *"what does this feature measure?"* — ask *"what does this feature tell us about the customer's behaviour?"* Every feature is designed to answer a specific business question, grouped into 5 perspectives on the same customer:

| Group | Business question | Features |
|-------|--------------------|----------|
| Purchase Rhythm | How often, and how predictably, do they buy? | `Recency`, `InterPurchaseCV`, `ActiveSpanDays`, `MonthlyOrderRate` |
| Spending Shape | How is their money spent? | `AOV`, `SpendGini`, `PriceCV`, `MaxSpendShare`, `SpendAcceleration` |
| Basket Behaviour | What do they buy each visit? | `NewSKURate`, `RepeatSKUFraction`, `BasketSizeCV`, `MedianBasketQty` |
| Volume & Bulk (B2B signal) | Is this actually a wholesale account? | `BurstIndex`, `BulkLineRate`, `QuantityCV`, `SKU_HHI` |
| Customer Lifecycle | What's invisible to RFM? | `ReturnRate`, `ReturnValueRate`, `QuarterConcentration` |

Notable feature definitions:
- **`InterPurchaseCV`** = std / mean of the gaps between purchases. Near 0 = buys on a predictable schedule; high = buys on impulse.
- **`SpendGini`** borrows the Gini coefficient from economics: near 0 means spend is evenly spread across invoices, ~0.8 means one or two invoices dominate total spend — same total spend, very different behaviour.
- **`SKU_HHI`** borrows the Herfindahl-Hirschman Index from antitrust economics: `Σ(revenue_SKU_i / total_revenue)²`. Near 1.0 means the customer buys almost exclusively one or two product types (a flower shop buying 80% from a single SKU scores ~0.64); near 0 means a diverse basket (~0.05 for a customer buying 20 different souvenirs). Values above 0.5 are treated as a strong B2B/wholesale signal — the single strongest signal in the whole feature set for that question.
- **`ReturnRate`** = cancelled invoices / total invoices — the single most important feature in the final model. `ReturnValueRate` (return value / total transaction value) was computed only to sanity-check `ReturnRate`, since the two correlate at Spearman 0.971, and was excluded from the model to avoid redundancy.
- **`QuarterConcentration`** = `clip(top quarter's spend / annual spend, 0.25, 1.0)` — the feature that operationalises the Q4-seasonality pattern found during EDA.

**Correlation filtering:** starting from 20 candidate features, pairs with `|r| > 0.85` in the model space (after scaling) were treated as redundant — since K-Means measures distance across all features equally, near-duplicate features would silently double-count that signal. 6 features were dropped, leaving **14 features** with a maximum remaining pairwise correlation of 0.781.

---

## Preprocessing Pipeline

Order matters — each step corrects a problem introduced (or left unaddressed) by the previous one:

1. **`QuantileTransformer(output='normal')`** — maps each feature's raw values to their percentile rank, turning skewed distributions into normal ones. Box-Cox was ruled out because many features are exact zero for a large share of customers (e.g. `ReturnRate` for anyone who has never returned an order), which Box-Cox can't handle.
2. **`StandardScaler`** - rescales everything to zero mean, unit variance. Necessary because K-Means measures distance with Euclidean distance: without this step, a feature measured in £ would dominate the distance calculation and drown out every other feature.
3. **Clip to ±3 standard deviations** — after scaling, a handful of B2B-like accounts had z-scores above 4. Clipping bounds their influence without deleting them from the dataset outright, keeping the information while limiting the damage extreme outliers do to centroid placement.

---

## Clustering Methodology

**Why K-Means** (over DBSCAN, Hierarchical, or Gaussian Mixture): simplicity, computational efficiency at this data size, and — critically for this use case — centroids that are directly interpretable by non-technical stakeholders as "the average behaviour of this segment."

**Why split off wholesale accounts before clustering, and why with a rule instead of ML:** rule-based separation is preferable to a learned model when the distinguishing signal is already clear-cut, domain knowledge is available to define the rule, and the class in question is too small a minority for a general-purpose model to learn reliably. All three hold here: 179 wholesale-like accounts (`SKU_HHI > 0.5`, 4.6% of customers) were separated by rule before clustering. Leaving them in degrades the retail clustering — Silhouette score is 0.177 on the mixed cohort versus **0.244** on the 3,741 retail-only customers once wholesale accounts are removed.

**Choosing k:** four metrics were tracked across k = 2–6 — Silhouette (↑ better), Davies-Bouldin (↓ better), Calinski-Harabasz (↑ better), and a weighted composite (`0.5×Silhouette + 0.25×1/(1+DB) + 0.25×CH/CH_max`, with Silhouette double-weighted since it most directly reflects cluster geometry). k=2 has the single highest Silhouette (0.315), but it only separates "buys a lot" from "buys a little" — too coarse to act on. k=4 (Silhouette 0.244, Davies-Bouldin 1.764, Calinski-Harabasz 1,066) was chosen over the statistically-optimal k=2 because going from k=3 to k=4 splits out a cluster of customers whose spend concentrates in Q4 — a segment that was previously blended into a larger cluster and that maps to a distinct, actionable marketing strategy. The final choice balances statistical quality against business actionability rather than optimising the metric alone.

---

## Key Results

**Two-stage pipeline:**
1. **Stage 1 (rule-based):** split off 179 wholesale-like accounts (`SKU_HHI > 0.5`) — 4.6% of the population
2. **Stage 2 (K-Means++, k=4):** clustered 3,741 retail customers on 14 behavioural features

**4 retail segments and their strategies:**

| Cluster | Name | n (%) | Key signal | Strategy |
|---------|------|-------|------------|----------|
| C0 | Seasonal Intensives | 330 (8.8%) | `QuarterConcentration` ↑, `MonthlyOrderRate` ↑ (Q4-heavy buying) | Pre-season campaign + a year-round loyalty program to soften seasonality. Don't judge them by Recency outside Q4. |
| C1 | High-Return Actives | 1,107 (29.6%) | `ReturnRate` ↑ — **absent from RFM**; still highly active | Improve product info (real photos, exact sizing). Don't penalise with return fees — they're active buyers still exploring the catalogue. |
| C2 | One-Time Explorers | 1,257 (33.6%) | `BasketSizeCV` ≈ 0, `ActiveSpanDays` ≈ 0, `NewSKURate` = 1.0 (no repeat SKUs) | Re-engage with category-based triggers tied to what they already bought — not generic discounts, since price isn't what's holding them back. |
| C3 | Occasional Loyalists | 1,047 (28.0%) | Zero returns, long tenure, moderate frequency, highest LTV | Light touch — exclusive access / early previews rather than discounts. Lowest support cost of any segment; don't over-contact them. |

---

## SHAP Explainability

**The problem:** K-Means only returns a cluster label (0–3) per customer — it can't say *why*. A marketing team needs to know whether a customer landed in C1 because of a high return rate or a short tenure, not just the label; and without knowing which features actually drive the cluster structure, there's no way to improve feature engineering going forward.

**The fix — a surrogate model:** train an explainable classifier (Random Forest, 200 trees) to reproduce the K-Means cluster assignments, then run SHAP (`TreeExplainer`) on that classifier. Cross-validated accuracy is 0.9904 ± 0.0023 — high enough that the Random Forest is a faithful stand-in for the K-Means boundaries, so its SHAP attributions can be trusted as an explanation of the clustering itself.

**Global feature importance** (share of total mean |SHAP| across all clusters):

| Feature | Importance |
|---------|-----------|
| `ReturnRate` | 27.5% |
| `QuarterConcentration` | 21.0% |
| `BasketSizeCV` | 18.4% |
| `RepeatSKUFraction` | 11.2% |
| `MonthlyOrderRate` | 10.0% |
| `InterPurchaseCV` | 5.7% |
| *(remaining 8 features)* | ~6.2% combined |

The top 3 features alone account for roughly two-thirds of total discriminating power — confirming that return behaviour, seasonal concentration, and basket-size consistency are what actually separate these segments, far more than raw RFM-style totals.

**Per-cluster direction** (positive SHAP = pushes a customer *into* that cluster): C1 is driven almost entirely by a positive `ReturnRate` signal; C0 is driven by positive `QuarterConcentration`; C2 by low `BasketSizeCV`/short tenure; C3 by the *absence* of the other clusters' driving signals (few returns, no strong seasonal concentration, stable basket size).

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
