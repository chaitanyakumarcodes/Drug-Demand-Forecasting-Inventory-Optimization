"""
Drug Demand Forecasting – Data Pipeline & ETL
=============================================
Loads raw Excel data, cleans it, and persists to a SQLite database.
"""

import sqlite3
import os
import pandas as pd
import numpy as np

RAW_PATH = os.path.join(os.path.dirname(__file__), "DSML_1feb.xlsx")
DB_PATH  = os.path.join(os.path.dirname(__file__), "drug_demand.db")


# ─────────────────────────────────────────────
# 1. EXTRACT
# ─────────────────────────────────────────────
def extract(path: str = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both sheets from the Excel file."""
    main   = pd.read_excel(path, sheet_name="Main")
    lookup = pd.read_excel(path, sheet_name="Lookup Data")
    print(f"[EXTRACT] Main: {main.shape}  |  Lookup: {lookup.shape}")
    return main, lookup


# ─────────────────────────────────────────────
# 2. TRANSFORM
# ─────────────────────────────────────────────
def _clean_age_group(val) -> str:
    """Standardise messy age_group codes (e.g. 'A!001' → 'A1001')."""
    if pd.isna(val):
        return "Unknown"
    val = str(val).strip().replace("!", "1")
    return val


def _sentiment_score(text: str) -> str:
    """
    Rule-based sentiment classification for news / press-release text.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    if pd.isna(text):
        return "Neutral"
    text = text.lower()
    positive_kw = ["approve", "launch", "success", "breakthrough", "growth",
                   "surge", "effective", "new", "available", "innovative"]
    negative_kw = ["shortage", "recall", "ban", "decline", "risk", "concern",
                   "failure", "adverse", "delay", "supply chain"]
    pos = sum(1 for k in positive_kw if k in text)
    neg = sum(1 for k in negative_kw if k in text)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"


def transform(main: pd.DataFrame) -> pd.DataFrame:
    """Clean, enrich, and engineer features on the main dataset."""
    df = main.copy()

    # ── Date / time features ──────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Year"]        = df["Date"].dt.year
    df["MonthNum"]    = df["Date"].dt.month
    df["DayOfYear"]   = df["Date"].dt.dayofyear
    df["Quarter"]     = df["Date"].dt.quarter
    df["WeekOfYear"]  = df["Date"].dt.isocalendar().week.astype(int)

    # ── Target: Demand ───────────────────────────────────────────
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
    # Clip extreme outliers (IQR-based, per drug) – more robust than 3-sigma
    def iqr_clip(series):
        q1, q3 = series.quantile(0.05), series.quantile(0.95)
        return series.clip(lower=0, upper=q3 + 3*(q3-q1))

    df["Demand"] = df.groupby("Drug")["Demand"].transform(iqr_clip)

    # ── Categorical clean-up ──────────────────────────────────────
    df["age_group"]  = df["age_group"].apply(_clean_age_group)
    df["Promotion"]  = df["Promotion"].astype(bool).astype(int)
    df["Region"]     = df["Region"].fillna("Unknown")
    df["Season"]     = df["Season"].fillna("Unknown")
    df["Drug_Type"]  = df["Drug_Type"].fillna("Unknown")
    df["Category"]   = df["Category"].fillna("Unknown")

    # ── NLP – sentiment from text columns ────────────────────────
    for col in ["Event", "Market_News", "Press_Release"]:
        df[f"{col}_sentiment"] = df[col].apply(_sentiment_score)

    # Aggregate sentiment score (simple majority vote)
    def majority_sentiment(row):
        vals = [row["Event_sentiment"], row["Market_News_sentiment"], row["Press_Release_sentiment"]]
        return max(set(vals), key=vals.count)
    df["Overall_Sentiment"] = df.apply(majority_sentiment, axis=1)

    # ── Lag features (per drug) ───────────────────────────────────
    df = df.sort_values(["Drug", "Date"])
    for lag in [1, 7, 14, 30]:
        df[f"Demand_lag_{lag}"] = df.groupby("Drug")["Demand"].shift(lag)

    df["Demand_rolling_7d"]  = df.groupby("Drug")["Demand"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["Demand_rolling_30d"] = df.groupby("Drug")["Demand"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean())

    # ── Fill NaN in lag cols ──────────────────────────────────────
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    df[lag_cols] = df[lag_cols].fillna(df["Demand"].median())

    # ── Prescription volume: fill missing ────────────────────────
    df["Prescription_Volume"] = pd.to_numeric(df["Prescription_Volume"], errors="coerce")
    df["Prescription_Volume"] = df["Prescription_Volume"].fillna(
        df.groupby("Drug")["Prescription_Volume"].transform("median"))

    print(f"[TRANSFORM] Output shape: {df.shape}  |  NaN count: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 3. LOAD
# ─────────────────────────────────────────────
def load(df: pd.DataFrame, db_path: str = DB_PATH):
    """Write cleaned data to SQLite and return connection."""
    conn = sqlite3.connect(db_path)

    # Persist transformed table
    df.to_sql("drug_demand", conn, if_exists="replace", index=False)

    # Aggregate views useful for inventory calculations
    agg = (df.groupby(["Drug", "Year", "MonthNum"])
             .agg(Total_Demand=("Demand","sum"),
                  Avg_Demand=("Demand","mean"),
                  Max_Demand=("Demand","max"),
                  Std_Demand=("Demand","std"),
                  Avg_Prescription=("Prescription_Volume","mean"))
             .reset_index())
    agg.to_sql("monthly_agg", conn, if_exists="replace", index=False)

    conn.commit()
    print(f"[LOAD] Written {len(df):,} rows → {db_path}")
    return conn


# ─────────────────────────────────────────────
# 4. RUN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(raw_path: str = RAW_PATH, db_path: str = DB_PATH) -> pd.DataFrame:
    main, lookup = extract(raw_path)
    df_clean     = transform(main)
    conn         = load(df_clean, db_path)
    conn.close()
    return df_clean


if __name__ == "__main__":
    df = run_pipeline()
    print(df.head(3).to_string())
    print("\nPipeline complete ✓")
