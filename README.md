# 💊 Drug Demand Forecasting & Inventory Optimization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-5EEAD4?style=for-the-badge"/>
</p>

<p align="center">
  <b>An end-to-end AI-powered pharmaceutical supply chain solution — forecasting drug demand, optimizing inventory, and extracting real-time signals from unstructured text.</b>
  <br/>
  <i>Digital Business Capability · Hackathon 2025 · Campus Hire</i>
</p>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [1. ETL Pipeline](#1-etl-pipeline)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. ML Modelling](#3-ml-modelling)
  - [4. NLP Analysis](#4-nlp-analysis)
  - [5. Inventory Optimization](#5-inventory-optimization)
- [Results](#-results)
- [Key Visuals](#-key-visuals)
- [Deliverables](#-deliverables)
- [Getting Started](#-getting-started)
- [Ethical Considerations](#-ethical-considerations)
- [Future Work](#-future-work)

---

## 🎯 Problem Statement

Pharmaceutical supply chains face two critical and opposing risks:

| Risk | Impact |
|---|---|
| **Stockouts** | Patient safety, treatment gaps, loss of trust |
| **Overstocking** | Drug expiry, capital waste, storage costs |

Manual demand planning is error-prone and ignores rich signals in unstructured data such as news articles, FDA announcements, and press releases. This project delivers a **data-driven, automated forecasting system** to eliminate both risks.

---

## 🧩 Solution Overview

The solution integrates four interconnected components into a single, reproducible pipeline:

```
Raw Excel Data
      │
      ▼
┌─────────────────┐     ┌──────────────────────┐
│   ETL Pipeline  │────▶│  Feature Engineering  │
│  data_pipeline  │     │  (Temporal + Lag +    │
│      .py        │     │   NLP Sentiment)      │
└─────────────────┘     └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      ML Model Training       │
                    │  Random Forest (Best)        │
                    │  Gradient Boosting           │
                    │  Ridge Regression (Baseline) │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐
     │  NLP Analysis  │  │  Inventory     │  │  Reports &       │
     │  nlp_analysis  │  │  Optimization  │  │  Visualisations  │
     │      .py       │  │  (SS, ROP,EOQ) │  │  (.docx, .pptx)  │
     └────────────────┘  └────────────────┘  └──────────────────┘
```

---

## 📁 Project Structure

```
drug-demand-forecasting/
│
├── 📄 data_pipeline.py              # ETL: Extract → Transform → Load to SQLite
├── 📄 model_training.py             # ML model training, evaluation, inventory calc
├── 📄 nlp_analysis.py               # NLP: sentiment, keyword extraction, demand drivers
├── 📄 drug_demand_forecasting.py    # Master notebook-style script (all stages end-to-end)
│
├── 📊 outputs/
│   ├── inventory_recommendations.csv      # 24 Drug × Region inventory parameters
│   ├── model_metrics.json                 # Benchmarked model results
│   ├── eda_demand_by_drug.png             # Monthly demand trends per drug
│   ├── eda_seasonality.png                # Seasonal demand distribution
│   ├── feature_importance.png             # Top 15 feature importances
│   ├── forecast_actual_vs_pred.png        # Actual vs Predicted demand
│   ├── model_comparison.png               # MAE / RMSE / R² comparison
│   ├── nlp_sentiment_demand.png           # Sentiment distribution & demand linkage
│   ├── nlp_demand_driver.png              # Boost/Reduce signals per drug
│   └── nlp_correlation.png                # Sentiment–Demand correlation heatmap
│
├── 📝 Drug_Demand_Forecasting_Report.docx       # Comprehensive 13-section report
├── 📊 Drug_Demand_Forecasting_Presentation.pptx # 10-slide stakeholder deck
├── 📋 requirements.txt                          # All Python dependencies
└── 📖 README.md                                 # This file
```

---

## 🏗️ Architecture

### Production Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│   ERP / POS Systems → News APIs → Press Releases                │
│              Azure Data Factory / AWS Glue                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        DATA STORAGE                             │
│   Azure Data Lake (raw)  →  PostgreSQL / SQLite (processed)     │
│              Feature Store (Feast) for ML features             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       ML PLATFORM                               │
│   Azure ML / SageMaker │ MLflow experiment tracking             │
│   Scheduled retraining │ Model versioning & registry            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     REAL-TIME NLP STREAM                        │
│   Apache Kafka → NLP Microservice → Sentiment Feature Store     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      SERVING LAYER                              │
│   FastAPI Inference Endpoints │ Power BI / Tableau Dashboards   │
│   Evidently AI Drift Monitor  │ Azure Monitor Alerts            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

The primary dataset (`DSML_1feb.xlsx`) contains **1,827 records** across two sheets:

| Feature | Type | Description |
|---|---|---|
| `Date` | DateTime | Transaction date (Jan 2020 – Jan 2025) |
| `Drug` | Categorical | 6 drugs: AID2, AmoxID1, AtoID3, CipID4, IIbuID5, ParID06 |
| `Demand` | Numeric (**Target**) | Units demanded per record |
| `Region` | Categorical | East, West, North, South |
| `Season` / `Month` | Categorical | Seasonality indicators |
| `Promotion` | Binary | Active promotional campaign (0/1) |
| `Prescription_Volume` | Numeric | Number of prescriptions issued |
| `age_group` / `Age` | Categorical / Numeric | Patient segment |
| `Drug_Type` / `Category` | Categorical | Drug classification codes |
| `Event` | Free Text | External events (NLP) |
| `Market_News` | Free Text | Market-level news (NLP) |
| `Press_Release` | Free Text | Company press releases (NLP) |

### ⚠️ Data Quality Issues Found & Fixed

| Issue | Fix Applied |
|---|---|
| Extreme outliers in `CipID4` (demand up to **10 million** units) | Robust IQR-based clipping (5th–95th pct + 3×IQR fence) |
| Negative demand values (physically impossible) | Clipped to zero |
| Malformed `age_group` codes (`A!001`) | Regex substitution (`!` → `1`) |
| Missing `Prescription_Volume` values (~220 rows) | Per-drug median imputation |
| 6 null `Demand` rows | Dropped before training |

---

## 🔬 Methodology

### 1. ETL Pipeline
**File:** `data_pipeline.py`

```python
# Run the pipeline
from data_pipeline import run_pipeline
df_clean = run_pipeline("DSML_1feb.xlsx", "drug_demand.db")
```

Three-stage pipeline:
- **Extract** — Loads both Excel sheets
- **Transform** — Cleans, clips outliers, engineers features, runs NLP sentiment
- **Load** — Persists cleaned data and monthly aggregates to SQLite

---

### 2. Feature Engineering

| Category | Features | Purpose |
|---|---|---|
| **Temporal** | Year, MonthNum, Quarter, WeekOfYear, DayOfYear, Weekday | Macro & micro time patterns |
| **Lag Demand** | Demand_lag_1, lag_7, lag_14, lag_30 | Short-term autocorrelation |
| **Rolling Mean** | Demand_rolling_7d, rolling_30d | Smoothed trend signal |
| **NLP Sentiment** | Event/Market_News/Press_Release sentiment + Overall_Sentiment | External demand signals |
| **Categorical** | Drug, Region, Season, Drug_Type, Category | Drug-level context |

> **Key insight:** Lag and rolling features account for **~65% of total feature importance**, confirming that recent demand history is the most powerful predictor.

---

### 3. ML Modelling
**File:** `model_training.py`

#### Train/Test Split Strategy
```
Timeline:  2020 ──────────────────────────── 2024-06-01 ──── 2025
                        TRAIN (88%)           │   TEST (12%)
```
A **time-based split** (no shuffling) prevents data leakage — exactly how real-world forecasting works.

#### Models Trained

```python
models = {
    "Random Forest":   RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5),
    "Gradient Boost":  GradientBoostingRegressor(n_estimators=200, lr=0.05, subsample=0.8),
    "Ridge Baseline":  Ridge(alpha=10.0),
}
```

---

### 4. NLP Analysis
**File:** `nlp_analysis.py`

Three-layer NLP approach — no external NLP libraries required:

```
Text Columns (Event, Market_News, Press_Release)
         │
         ▼
┌─────────────────────────────┐
│  Rule-Based Keyword Scoring │  →  Positive / Negative / Neutral
│  (Positive & Negative KW)   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Demand Driver Detection    │  →  Boost / Reduce / Neutral
│  (Boost & Reduce KW)        │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  TF-IDF Bigram Extraction   │  →  Top demand-influencing phrases
│  (sklearn TfidfVectorizer)  │     per drug
└─────────────────────────────┘
```

---

### 5. Inventory Optimization

Using the best model's forecast as input, inventory parameters are computed per **Drug × Region** combination:

| Parameter | Formula | Parameters Used |
|---|---|---|
| **Safety Stock** | `Z × σ_demand × √(Lead Time)` | Z = 1.65 (95% service level), LT = 7 days |
| **Reorder Point** | `Avg Demand × Lead Time + Safety Stock` | — |
| **Optimal Order Qty** | `Avg Daily Demand × 14` | 2-week EOQ approximation |

```python
# Run inventory calculations
from model_training import compute_inventory
inv_df = compute_inventory(test_predictions, service_level_z=1.65)
inv_df.to_csv("inventory_recommendations.csv", index=False)
```

---

## 📊 Results

### Model Benchmark

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| 🏆 **Random Forest** | **19.69** | **39.24** | **0.1823** | **29.18%** |
| Gradient Boosting | 35.30 | 56.86 | -0.717 | 52.01% |
| Ridge (Baseline) | 19.43 | 39.61 | 0.1668 | 30.12% |

> ✅ **Random Forest** selected as production model — lowest RMSE and highest R².

### NLP Results

| Metric | Value |
|---|---|
| Positive sentiment records | 91.7% (1,675 / 1,827) |
| Boost demand-signal records | 71.3% (1,301 / 1,827) |
| Top bigrams | `pain reliever`, `launch new drug`, `clinical trial`, `FDA approve`, `antibiotic demand` |
| Demand uplift on negative sentiment | ~2–5% local spike |

### Inventory Highlights (Top Priority SKUs)

| Drug | Region | Avg Daily Demand | Safety Stock | Reorder Point | Order Qty |
|---|---|---|---|---|---|
| ParID06 | North | 417.3 | 4,497 | 7,418 | 5,842 |
| CipID4 | East | 342.7 | 2,068 | 4,467 | 4,798 |
| CipID4 | West | 277.3 | 3,194 | 5,135 | 3,882 |
| AmoxID1 | South | 214.0 | 2,108 | 3,606 | 2,996 |

*Full 24-row table in `outputs/inventory_recommendations.csv`*

---

## 🖼️ Key Visuals

<table>
  <tr>
    <td align="center"><b>Monthly Demand by Drug</b></td>
    <td align="center"><b>Model Comparison</b></td>
  </tr>
  <tr>
    <td><code>outputs/eda_demand_by_drug.png</code></td>
    <td><code>outputs/model_comparison.png</code></td>
  </tr>
  <tr>
    <td align="center"><b>Top 15 Feature Importances</b></td>
    <td align="center"><b>Actual vs Predicted</b></td>
  </tr>
  <tr>
    <td><code>outputs/feature_importance.png</code></td>
    <td><code>outputs/forecast_actual_vs_pred.png</code></td>
  </tr>
  <tr>
    <td align="center"><b>Sentiment Distribution</b></td>
    <td align="center"><b>Demand Driver by Drug</b></td>
  </tr>
  <tr>
    <td><code>outputs/nlp_sentiment_demand.png</code></td>
    <td><code>outputs/nlp_demand_driver.png</code></td>
  </tr>
</table>

---

## 📋 Deliverables

| # | File | Description |
|---|---|---|
| 1 | `Drug_Demand_Forecasting_Report.docx` | 13-section comprehensive report (EDA → Modelling → NLP → Inventory → Deployment → Ethics) |
| 2 | `Drug_Demand_Forecasting_Presentation.pptx` | 10-slide stakeholder presentation with embedded charts |
| 3 | `drug_demand_forecasting.py` | Master end-to-end script (all 8 workflow stages, Jupyter-compatible) |
| 4 | `data_pipeline.py` | Standalone ETL module |
| 5 | `model_training.py` | Standalone ML training & evaluation module |
| 6 | `nlp_analysis.py` | Standalone NLP analysis module |
| 7 | `inventory_recommendations.csv` | 24 Drug × Region inventory parameters |
| 8 | `model_metrics.json` | Benchmarked model results (MAE, RMSE, R², MAPE) |
| 9 | `requirements.txt` | Full dependency list |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- The raw data file: `DSML_1feb.xlsx`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/drug-demand-forecasting.git
cd drug-demand-forecasting

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your data file in the project root
cp /path/to/DSML_1feb.xlsx .
```

### Run the Full Pipeline

```bash
# Option A: Run everything end-to-end
python drug_demand_forecasting.py

# Option B: Run each stage independently
python data_pipeline.py       # Step 1: ETL
python model_training.py      # Step 2: ML Training + Inventory
python nlp_analysis.py        # Step 3: NLP Analysis
```

### Expected Outputs

After running, the `outputs/` directory will contain:

```
outputs/
├── inventory_recommendations.csv   # Inventory parameters per Drug × Region
├── model_metrics.json              # Model benchmark results
├── eda_demand_by_drug.png          # EDA visualisation
├── eda_seasonality.png             # Seasonality chart
├── feature_importance.png          # RF feature importance
├── forecast_actual_vs_pred.png     # Forecast accuracy chart
├── model_comparison.png            # Model benchmark comparison
├── nlp_sentiment_demand.png        # Sentiment analysis chart
├── nlp_demand_driver.png           # Demand driver analysis
└── nlp_correlation.png             # Sentiment-demand correlation
```

### Quick Module Usage

```python
# ETL
from data_pipeline import run_pipeline
df = run_pipeline("DSML_1feb.xlsx", "drug_demand.db")

# Sentiment scoring (no external libraries needed)
from nlp_analysis import sentiment_score, detect_demand_driver
score, label = sentiment_score("FDA approves new antibiotic for resistant infections")
# → (0.5, 'Positive')

driver = detect_demand_driver("Surge in antibiotic demand due to outbreak")
# → 'Boost'

# Inventory parameters
from model_training import compute_inventory
inv_df = compute_inventory(test_predictions_df, service_level_z=1.65)
```

---

## ⚖️ Ethical Considerations

| Area | Risk | Mitigation |
|---|---|---|
| **Demographic Bias** | `age_group` underrepresentation | Stratified error monitoring by patient segment |
| **Promotion Bias** | Past campaigns inflate historical demand | `Promotion` explicitly included as a feature; evaluate on promotion-free periods |
| **Geographic Imbalance** | Sparse data in some regions | Regional smoothing; flag low-confidence predictions |
| **Data Privacy** | Patient sensitivity | No PII stored; only aggregate demand data processed |
| **Responsible AI** | Black-box predictions | SHAP feature importances provided for every prediction |
| **Human-in-the-Loop** | Automation risk | All inventory decisions are **recommendations** requiring pharmacist approval |

---

## 🔮 Future Work

- [ ] **Time-Series Models** — Add ARIMA / SARIMA / Facebook Prophet as benchmarks
- [ ] **Advanced NLP** — Replace rule-based sentiment with BioBERT / PharmBERT
- [ ] **Hierarchical Forecasting** — National → Regional → Drug decomposition
- [ ] **Online Learning** — Incremental model retraining on rolling windows
- [ ] **SHAP Explainability Dashboard** — Waterfall plots per prediction for pharmacists
- [ ] **Cloud Deployment** — Azure ML pipeline with Kafka streaming and FastAPI serving
- [ ] **A/B Testing Framework** — Evaluate model updates before full rollout
- [ ] **External Data Sources** — Disease incidence rates, competitor pricing, regulatory calendars

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest, Gradient Boosting, Ridge) |
| NLP | scikit-learn TF-IDF, Rule-based keyword matching |
| Visualisation | Matplotlib, Seaborn |
| Database | SQLite (dev/test), PostgreSQL (production) |
| Report Generation | python-docx (Word), pptxgenjs (PowerPoint) |
| Deployment (proposed) | FastAPI, Docker, Kubernetes, Azure ML / AWS SageMaker |
| Monitoring (proposed) | MLflow, Evidently AI, Apache Kafka |

---

## 📄 References

1. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
2. Breiman, L. (2001). *Random Forests*. Machine Learning 45, 5–32.
3. Friedman, J.H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics 29(5), 1189–1232.
4. Silver, E.A., Pyke, D.F. & Thomas, D.J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.

---

<p align="center">
  Made with ❤️ for the Digital Business Capability Hackathon 2025
  <br/>
  <i>End-to-end solution: ETL · Machine Learning · NLP · Inventory Optimization · Reporting</i>
</p>
