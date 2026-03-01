# %% [markdown]
# # Drug Demand Forecasting & Inventory Optimization
# ## End-to-End Jupyter Notebook Workflow
# **Digital Business Capability Hackathon – Campus Hire 2025**
#
# This notebook covers the complete workflow:
# 1. Data Loading & ETL
# 2. Data Understanding & Preprocessing
# 3. Exploratory Data Analysis (EDA)
# 4. Feature Engineering
# 5. Model Training & Evaluation (Random Forest vs Gradient Boosting)
# 6. NLP on Social/Event Text
# 7. Inventory Optimization
# 8. Results & Recommendations

# %% [markdown]
# ---
# ## 1. Setup & Imports

# %%
import os
import sys
import sqlite3
import warnings
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "#F8FAFC", "axes.facecolor": "#F8FAFC",
                      "axes.grid": True, "grid.alpha": 0.3})

# Paths
RAW_PATH  = "DSML_1feb.xlsx"
DB_PATH   = "drug_demand.db"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("✓ Imports done | pandas", pd.__version__, "| sklearn", __import__("sklearn").__version__)


# %% [markdown]
# ---
# ## 2. Data Loading & ETL Pipeline

# %%
# --- EXTRACT ---
df_main   = pd.read_excel(RAW_PATH, sheet_name="Main")
df_lookup = pd.read_excel(RAW_PATH, sheet_name="Lookup Data")
print(f"Main: {df_main.shape}  |  Lookup: {df_lookup.shape}")
df_main.head(3)


# %%
# --- TRANSFORM ---
def iqr_clip(series):
    """Robust IQR-based clipping: 5th pct to Q3 + 3×IQR."""
    q1, q3 = series.quantile(0.05), series.quantile(0.95)
    return series.clip(lower=0, upper=q3 + 3*(q3 - q1))

def clean_age_group(val):
    if pd.isna(val): return "Unknown"
    return str(val).strip().replace("!", "1")

def sentiment_rule(text):
    if pd.isna(text): return 0.0, "Neutral"
    pos_kw = ["approve","launch","success","breakthrough","growth","surge",
               "effective","new","available","recommend","relief"]
    neg_kw = ["shortage","recall","ban","decline","risk","failure",
               "adverse","delay","warning","discontinue"]
    pos = sum(1 for k in pos_kw if k in text.lower())
    neg = sum(1 for k in neg_kw if k in text.lower())
    score = (pos - neg) / max(pos + neg, 1)
    label = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
    return score, label

df = df_main.copy()

# Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Target
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
df["Demand"] = df.groupby("Drug")["Demand"].transform(iqr_clip)

# Categorical
df["age_group"] = df["age_group"].apply(clean_age_group)
df["Promotion"] = df["Promotion"].astype(bool).astype(int)
df["Prescription_Volume"] = pd.to_numeric(df["Prescription_Volume"], errors="coerce")
df["Prescription_Volume"] = df["Prescription_Volume"].fillna(
    df.groupby("Drug")["Prescription_Volume"].transform("median"))

# Time features
df["Year"]       = df["Date"].dt.year
df["MonthNum"]   = df["Date"].dt.month
df["DayOfYear"]  = df["Date"].dt.dayofyear
df["Quarter"]    = df["Date"].dt.quarter
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# NLP
for col in ["Event", "Market_News", "Press_Release"]:
    df[[f"{col}_score", f"{col}_sentiment"]] = df[col].apply(
        lambda t: pd.Series(sentiment_rule(t)))

def majority_sentiment(row):
    vals = [row[f"{c}_sentiment"] for c in ["Event","Market_News","Press_Release"]]
    return max(set(vals), key=vals.count)
df["Overall_Sentiment"] = df.apply(majority_sentiment, axis=1)

# Lag features
df = df.sort_values(["Drug", "Date"])
for lag in [1, 7, 14, 30]:
    df[f"Demand_lag_{lag}"] = df.groupby("Drug")["Demand"].shift(lag)
df["Demand_rolling_7d"]  = df.groupby("Drug")["Demand"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df["Demand_rolling_30d"] = df.groupby("Drug")["Demand"].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean())

# Fill lag NaN with median
lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
df[lag_cols] = df[lag_cols].fillna(df["Demand"].median())

print(f"✓ Transform complete: {df.shape}  |  NaN: {df.isnull().sum().sum()}")


# %%
# --- LOAD to SQLite ---
conn = sqlite3.connect(DB_PATH)
df.to_sql("drug_demand", conn, if_exists="replace", index=False)
agg = (df.groupby(["Drug", "Year", "MonthNum"])
         .agg(Total_Demand=("Demand","sum"), Avg_Demand=("Demand","mean"),
              Max_Demand=("Demand","max"), Std_Demand=("Demand","std"))
         .reset_index())
agg.to_sql("monthly_agg", conn, if_exists="replace", index=False)
conn.commit(); conn.close()
print(f"✓ Loaded {len(df):,} rows to SQLite: {DB_PATH}")


# %% [markdown]
# ---
# ## 3. Exploratory Data Analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Drug Demand Forecasting – EDA Dashboard", fontsize=14, fontweight="bold")

# 3.1 Demand distribution per drug
df.boxplot(column="Demand", by="Drug", ax=axes[0,0], patch_artist=True)
axes[0,0].set_title("Demand Distribution by Drug"); axes[0,0].set_xlabel("Drug")
plt.sca(axes[0,0]); plt.xticks(rotation=30)

# 3.2 Monthly average demand
monthly = df.groupby(["MonthNum","Drug"])["Demand"].mean().unstack()
monthly.plot(ax=axes[0,1], linewidth=1.5)
axes[0,1].set_title("Average Demand by Month and Drug")
axes[0,1].set_xlabel("Month Number"); axes[0,1].legend(fontsize=8)

# 3.3 Demand by Season
season_order = ["Winter","Spring","Summer","Fall"]
sdf = [df[df["Season"]==s]["Demand"].dropna() for s in season_order if s in df["Season"].unique()]
axes[1,0].boxplot(sdf, tick_labels=[s for s in season_order if s in df["Season"].unique()],
                  patch_artist=True)
axes[1,0].set_title("Demand by Season")

# 3.4 Demand by Region
region_avg = df.groupby("Region")["Demand"].mean()
axes[1,1].bar(region_avg.index, region_avg.values, color=["#065A82","#1C7293","#5EEAD4","#F96167"])
axes[1,1].set_title("Average Demand by Region"); axes[1,1].set_ylabel("Avg Demand")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_dashboard.png", dpi=120, bbox_inches="tight")
plt.show()
print("EDA statistics:")
print(df.groupby("Drug")["Demand"].describe().round(2))


# %% [markdown]
# ---
# ## 4. Feature Engineering Summary

# %%
CATEGORICAL_FEATURES = ["Drug", "Region", "Season", "Drug_Type",
                         "Overall_Sentiment", "Category"]
NUMERIC_FEATURES     = ["Year", "MonthNum", "DayOfYear", "Quarter",
                         "WeekOfYear", "Weekday", "Promotion", "Age",
                         "Prescription_Volume",
                         "Demand_lag_1", "Demand_lag_7", "Demand_lag_14",
                         "Demand_lag_30", "Demand_rolling_7d", "Demand_rolling_30d"]
TARGET = "Demand"

print(f"Total features: {len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)}")
print(f"  Categorical: {CATEGORICAL_FEATURES}")
print(f"  Numeric: {NUMERIC_FEATURES}")


# %% [markdown]
# ---
# ## 5. Model Training & Evaluation

# %%
# Time-based train/test split (no shuffling – avoids leakage)
CUTOFF = pd.Timestamp("2024-06-01")
train  = df[df["Date"] < CUTOFF].copy()
test   = df[df["Date"] >= CUTOFF].copy()
print(f"Train: {len(train):,}  |  Test: {len(test):,}")

# Encode
def prepare_data(data, encoders=None):
    data = data.copy()
    encoders = encoders or {}
    for col in CATEGORICAL_FEATURES:
        le = encoders.get(col, LabelEncoder())
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    all_feats = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = data[all_feats].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = pd.to_numeric(data[TARGET], errors="coerce").fillna(data[TARGET].median())
    return X, y, encoders

X_train, y_train, encoders = prepare_data(train)
X_test,  y_test,  _        = prepare_data(test, encoders)


# %%
# Model 1: Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                  min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = np.clip(rf_model.predict(X_test), 0, None)

# Model 2: Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = np.clip(gb_model.predict(X_test), 0, None)

# Baseline: Ridge
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)
ridge  = Ridge(alpha=10.0)
ridge.fit(X_tr_s, y_train)
ri_pred = ridge.predict(X_te_s)

def eval_model(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    print(f"[{name:22s}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": round(mae,2), "RMSE": round(rmse,2),
            "R2": round(r2,4), "MAPE%": round(mape,2)}

print("\n── Model Evaluation ─────────────────────────────────")
results = [
    eval_model(y_test, rf_pred, "Random Forest"),
    eval_model(y_test, gb_pred, "Gradient Boosting"),
    eval_model(y_test, ri_pred, "Ridge (Baseline)"),
]
results_df = pd.DataFrame(results)
print("\nComparison table:")
print(results_df.to_string(index=False))


# %%
# Feature Importance Plot
feat_names = CATEGORICAL_FEATURES + NUMERIC_FEATURES
imp = pd.Series(rf_model.feature_importances_, index=feat_names).nlargest(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(imp.index[::-1], imp.values[::-1], color="#065A82")
axes[0].set_title("Top 15 Feature Importances – Random Forest", fontweight="bold")
axes[0].set_xlabel("Importance")

# Actual vs Predicted for first drug
drug_test = test.copy()
drug_test["Predicted"] = rf_pred
for drug in test["Drug"].unique()[:1]:
    sub = drug_test[drug_test["Drug"]==drug].sort_values("Date")
    axes[1].plot(sub["Date"], sub["Demand"],   label="Actual",    linewidth=1.5, color="#065A82")
    axes[1].plot(sub["Date"], sub["Predicted"],label="Predicted", linewidth=1.5,
                 linestyle="--", color="#5EEAD4")
    axes[1].set_title(f"Actual vs Predicted – {drug}", fontweight="bold")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_results.png", dpi=120, bbox_inches="tight")
plt.show()


# %% [markdown]
# ---
# ## 6. NLP Analysis

# %%
BOOST_KW  = ["surge","increase","approve","launch","outbreak","epidemic","new drug","growing"]
REDUCE_KW = ["recall","shortage","ban","decline","failure","adverse","supply chain","discontinue"]

def get_driver(text):
    if pd.isna(text): return "Neutral"
    t = text.lower()
    b = sum(1 for k in BOOST_KW  if k in t)
    r = sum(1 for k in REDUCE_KW if k in t)
    return "Boost" if b > r else ("Reduce" if r > b else "Neutral")

df["Demand_Driver"] = df["Event"].apply(get_driver)

print("Sentiment Distribution:")
print(df["Overall_Sentiment"].value_counts())
print("\nDemand Driver Distribution:")
print(df["Demand_Driver"].value_counts())
print("\nAvg Demand by Sentiment:")
print(df.groupby("Overall_Sentiment")["Demand"].mean().round(2))


# %%
# TF-IDF Keyword Extraction
corpus = (df["Event"].fillna("") + " " + df["Market_News"].fillna("")).tolist()
vec = TfidfVectorizer(ngram_range=(2,2), max_features=50, stop_words="english",
                      token_pattern=r"[a-zA-Z]{3,}")
X_tfidf = vec.fit_transform(corpus)
scores  = X_tfidf.mean(axis=0).A1
vocab   = vec.get_feature_names_out()
top20   = sorted(zip(vocab, scores), key=lambda x: -x[1])[:20]
print("\nTop 20 Demand-Influencing Bigrams:")
for phrase, score in top20:
    print(f"  {phrase:<30} {score:.4f}")


# %% [markdown]
# ---
# ## 7. Inventory Optimization

# %%
test_with_pred = test.copy()
test_with_pred["Predicted_Demand"] = rf_pred

Z = 1.65   # 95th pct service level
LT = 7     # 7-day lead time

inventory = []
for (drug, region), grp in test_with_pred.groupby(["Drug","Region"]):
    avg  = grp["Predicted_Demand"].mean()
    std  = grp["Predicted_Demand"].std() or 0
    ss   = Z * std * np.sqrt(LT)
    rop  = avg * LT + ss
    ooq  = avg * 14    # 2-week supply
    inventory.append({
        "Drug": drug, "Region": region,
        "Avg_Daily_Demand": round(avg, 2),
        "Std_Daily_Demand": round(std, 2),
        "Safety_Stock":     round(ss, 0),
        "Reorder_Point":    round(rop, 0),
        "Optimal_Order_Qty":round(ooq, 0),
    })

inv_df = pd.DataFrame(inventory)
inv_df.to_csv(f"{OUT_DIR}/inventory_recommendations.csv", index=False)
print("Inventory Recommendations:")
print(inv_df.sort_values("Avg_Daily_Demand", ascending=False).head(10).to_string(index=False))


# %% [markdown]
# ---
# ## 8. Summary

# %%
print("=" * 60)
print("DRUG DEMAND FORECASTING – SUMMARY")
print("=" * 60)
print(f"\nDataset: {df.shape[0]:,} records | {df['Drug'].nunique()} drugs | {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"\nBest Model: Random Forest")
best = next(r for r in results if r["model"] == "Random Forest")
print(f"  MAE  = {best['MAE']}")
print(f"  RMSE = {best['RMSE']}")
print(f"  R²   = {best['R2']}")
print(f"  MAPE = {best['MAPE%']}%")
print(f"\nInventory Recommendations: {len(inv_df)} Drug × Region combinations")
print(f"  Highest priority: {inv_df.loc[inv_df['Reorder_Point'].idxmax(), 'Drug']} in {inv_df.loc[inv_df['Reorder_Point'].idxmax(), 'Region']}")
print(f"    Reorder Point = {inv_df['Reorder_Point'].max():,.0f} units")
print(f"\nNLP Sentiment: {df['Overall_Sentiment'].value_counts().to_dict()}")
print(f"\n✓ All outputs written to ./{OUT_DIR}/")
