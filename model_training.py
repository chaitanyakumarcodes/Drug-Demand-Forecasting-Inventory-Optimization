"""
Drug Demand Forecasting – Model Training & Evaluation
=====================================================
Implements two models:
  1. Random Forest Regressor
  2. Gradient Boosting Regressor
Uses time-series aware train/test split (no data leakage).
Outputs metrics, feature importances, and inventory recommendations.
"""

import os
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from data_pipeline import run_pipeline

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
TRAIN_CUTOFF = "2024-06-01"   # train on data before this date
OUTPUT_DIR   = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORICAL_FEATURES = ["Drug", "Region", "Season", "Drug_Type",
                         "Overall_Sentiment", "Category"]
NUMERIC_FEATURES     = ["Year", "MonthNum", "DayOfYear", "Quarter",
                         "WeekOfYear", "Weekday", "Promotion", "Age",
                         "Prescription_Volume",
                         "Demand_lag_1", "Demand_lag_7", "Demand_lag_14",
                         "Demand_lag_30", "Demand_rolling_7d", "Demand_rolling_30d"]
TARGET               = "Demand"

MODELS = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12,
                                           min_samples_leaf=5, n_jobs=-1,
                                           random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                    learning_rate=0.05,
                                                    subsample=0.8, random_state=42),
}


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None
                         ) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Fit or reuse encoders."""
    df = df.copy()
    encoders = encoders or {}
    for col in CATEGORICAL_FEATURES:
        le = encoders.get(col, LabelEncoder())
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def prepare_features(df: pd.DataFrame, encoders: dict | None = None
                     ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Encode, select features, return X, y, encoders."""
    df_enc, enc = encode_categoricals(df, encoders)
    all_feats = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df_enc[all_feats].copy()
    # Force all to numeric (label-encoded cats are already int; coerce any residuals)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    y = df_enc[TARGET].fillna(df_enc[TARGET].median())
    return X, y, enc


def metrics(y_true, y_pred, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    print(f"  [{label}]  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(model=label, MAE=round(mae,2), RMSE=round(rmse,2),
                R2=round(r2,4), MAPE_pct=round(mape,2))


# ─────────────────────────────────────────────────
# Inventory Optimization
# ─────────────────────────────────────────────────
def compute_inventory(df: pd.DataFrame, forecast_col: str = "Predicted_Demand",
                      service_level_z: float = 1.65) -> pd.DataFrame:
    """
    For each Drug × Region compute:
      - Average Daily Demand
      - Std Dev of Demand
      - Safety Stock  = Z * std * sqrt(lead_time)
      - Reorder Point = avg_demand * lead_time + safety_stock
      - Optimal Order Qty (EOQ-inspired: 2 weeks of avg demand)
    """
    LEAD_TIME_DAYS = 7  # assumed lead time
    grp = df.groupby(["Drug", "Region"])[forecast_col]
    avg  = grp.mean()
    std  = grp.std().fillna(0)
    inv  = pd.DataFrame({
        "Avg_Daily_Demand": avg.round(2),
        "Std_Daily_Demand": std.round(2),
        "Safety_Stock":     (service_level_z * std * np.sqrt(LEAD_TIME_DAYS)).round(0),
        "Reorder_Point":    (avg * LEAD_TIME_DAYS + service_level_z * std * np.sqrt(LEAD_TIME_DAYS)).round(0),
        "Optimal_Order_Qty": (avg * 14).round(0),   # 2-week supply
    }).reset_index()
    return inv


# ─────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────
PALETTE = ["#065A82", "#1C7293", "#5EEAD4", "#F96167"]

def plot_actual_vs_predicted(df_test: pd.DataFrame, model_name: str, fname: str):
    drugs = df_test["Drug_orig"].unique()[:3]
    fig, axes = plt.subplots(len(drugs), 1, figsize=(12, 4*len(drugs)),
                             facecolor="#F8FAFC")
    for ax, drug in zip(np.atleast_1d(axes), drugs):
        sub = df_test[df_test["Drug_orig"] == drug].sort_values("Date")
        ax.plot(sub["Date"], sub["Demand"], color=PALETTE[0], linewidth=1.5,
                label="Actual")
        ax.plot(sub["Date"], sub["Predicted_Demand"], color=PALETTE[2],
                linewidth=1.5, linestyle="--", label="Predicted")
        ax.set_title(f"{drug} — {model_name}", fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3); ax.set_facecolor("#F8FAFC")
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str, fname: str):
    imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(15)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F8FAFC")
    bars = ax.barh(imp.index[::-1], imp.values[::-1], color=PALETTE[1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=13,
                 fontweight="bold")
    ax.set_facecolor("#F8FAFC"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results: list, fname: str):
    df_r  = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#F8FAFC")
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(df_r["model"], df_r[metric], color=PALETTE[:len(df_r)], width=0.5)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_facecolor("#F8FAFC"); ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(df_r[metric]):
            ax.text(i, v * 1.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_demand_by_drug(df: pd.DataFrame, fname: str):
    """Monthly average demand per drug."""
    df["YearMonth"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
    pivot = df.groupby(["YearMonth", "Drug"])["Demand"].mean().unstack("Drug")
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#F8FAFC")
    colors = plt.cm.tab10.colors
    for i, col in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[col], label=col, color=colors[i], linewidth=1.8)
    ax.set_xlabel("Month"); ax.set_ylabel("Average Demand")
    ax.set_title("Monthly Average Demand by Drug", fontsize=13, fontweight="bold")
    step = max(1, len(pivot) // 12)
    ax.set_xticks(range(0, len(pivot), step))
    ax.set_xticklabels(pivot.index[::step], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_facecolor("#F8FAFC")
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_seasonality(df: pd.DataFrame, fname: str):
    """Box plot of demand by season."""
    order = ["Spring", "Summer", "Fall", "Autumn", "Winter"]
    seasons = [s for s in order if s in df["Season"].unique()]
    data_by_season = [df[df["Season"]==s]["Demand"].dropna().values for s in seasons]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F8FAFC")
    bp = ax.boxplot(data_by_season, labels=seasons, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
    ax.set_ylabel("Demand"); ax.set_title("Demand Distribution by Season",
                                          fontsize=13, fontweight="bold")
    ax.set_facecolor("#F8FAFC"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Drug Demand Forecasting — Model Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    db_path = "/home/claude/drug_demand.db"
    if not os.path.exists(db_path):
        print("Running pipeline first …")
        df_full = run_pipeline()
    else:
        conn   = sqlite3.connect(db_path)
        df_full = pd.read_sql("SELECT * FROM drug_demand", conn, parse_dates=["Date"])
        conn.close()

    df_full["Date"] = pd.to_datetime(df_full["Date"])
    print(f"Dataset: {df_full.shape[0]:,} rows | Drugs: {df_full['Drug'].nunique()}")

    # ── EDA plots ────────────────────────────────────────────────
    print("\n[EDA] Generating plots …")
    plot_demand_by_drug(df_full, f"{OUTPUT_DIR}/eda_demand_by_drug.png")
    plot_seasonality(df_full,    f"{OUTPUT_DIR}/eda_seasonality.png")

    # ── Train / Test split ───────────────────────────────────────
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train  = df_full[df_full["Date"] < cutoff].copy()
    test   = df_full[df_full["Date"] >= cutoff].copy()
    print(f"\nTrain: {len(train):,} rows  |  Test: {len(test):,} rows")

    # Preserve original Drug label for plotting
    test["Drug_orig"] = test["Drug"].copy()

    X_train, y_train, encoders = prepare_features(train)
    X_test,  y_test,  _        = prepare_features(test, encoders)

    # ── Train models ─────────────────────────────────────────────
    results   = []
    all_preds = {}
    trained   = {}

    print("\nTraining models:")
    for name, model in MODELS.items():
        print(f"  Fitting {name} …")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)   # demand can't be negative
        all_preds[name] = preds
        trained[name]   = model
        results.append(metrics(y_test, preds, name))

    # ── Also fit Ridge as baseline ────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    ridge  = Ridge(alpha=10.0)
    ridge.fit(X_tr_s, y_train)
    ridge_pred = ridge.predict(X_te_s)
    results.append(metrics(y_test, ridge_pred, "Ridge (baseline)"))

    # ── Best model ───────────────────────────────────────────────
    best_name = min((r for r in results if r["model"] != "Ridge (baseline)"),
                    key=lambda r: r["RMSE"])["model"]
    best_model = trained[best_name]
    best_preds = all_preds[best_name]
    print(f"\n✓ Best model: {best_name}")

    # ── Plots ────────────────────────────────────────────────────
    print("\n[PLOTS] Generating …")
    test_with_pred = test.copy()
    test_with_pred["Predicted_Demand"] = best_preds

    plot_actual_vs_predicted(test_with_pred, best_name,
                             f"{OUTPUT_DIR}/forecast_actual_vs_pred.png")
    plot_feature_importance(best_model,
                            CATEGORICAL_FEATURES + NUMERIC_FEATURES,
                            best_name,
                            f"{OUTPUT_DIR}/feature_importance.png")
    plot_model_comparison(results, f"{OUTPUT_DIR}/model_comparison.png")

    # ── Inventory Optimization ────────────────────────────────────
    print("\n[INVENTORY] Computing optimal inventory levels …")
    inv_df = compute_inventory(test_with_pred, forecast_col="Predicted_Demand")
    inv_path = f"{OUTPUT_DIR}/inventory_recommendations.csv"
    inv_df.to_csv(inv_path, index=False)
    print(inv_df.to_string(index=False))

    # ── Save metrics ──────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ All outputs saved to", OUTPUT_DIR)
    return results, inv_df


if __name__ == "__main__":
    main()
