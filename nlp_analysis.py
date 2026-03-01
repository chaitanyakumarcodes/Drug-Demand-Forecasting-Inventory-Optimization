"""
Drug Demand Forecasting – NLP Analysis on Social/Event Text
============================================================
Analyses Event, Market_News, Press_Release columns using:
  1. Rule-based keyword sentiment classification
  2. TF-IDF keyword extraction per drug
  3. Demand driver identification (Boost vs. Reduce signals)
  4. Correlation between sentiment and demand
"""

import os
import sqlite3
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

DB_PATH    = "/home/claude/drug_demand.db"
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ["#065A82", "#1C7293", "#5EEAD4", "#F96167", "#F9E795", "#028090"]

# ─────────────────────────────────────────────────────────────────
# Keyword dictionaries
# ─────────────────────────────────────────────────────────────────
DEMAND_BOOST_KW = [
    "surge", "increase", "approve", "launch", "success", "breakthrough",
    "new drug", "high demand", "growing", "effective", "available", "recommend",
    "prescribe", "outbreak", "epidemic", "pandemic", "infection", "approve"
]
DEMAND_REDUCE_KW = [
    "recall", "shortage", "ban", "decline", "risk", "concern", "failure",
    "adverse", "delay", "supply chain", "safety", "warning", "discontinue",
    "alternative", "generic", "cheap"
]
POSITIVE_KW = [
    "approve", "launch", "success", "breakthrough", "growth", "surge",
    "effective", "new", "available", "innovative", "recommend", "relief",
    "treatment", "therapy", "clinical", "trial"
]
NEGATIVE_KW = [
    "shortage", "recall", "ban", "decline", "risk", "concern", "failure",
    "adverse", "delay", "warning", "safety", "discontinue", "expire"
]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def simple_tokenize(text: str) -> list[str]:
    """Lowercase and tokenize without external NLTK dependency."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def sentiment_score(text: str) -> tuple[float, str]:
    """Return (+1 positive, -1 negative, 0 neutral) and label."""
    if pd.isna(text):
        return 0.0, "Neutral"
    tokens = simple_tokenize(text)
    pos = sum(1 for k in POSITIVE_KW if k in text.lower())
    neg = sum(1 for k in NEGATIVE_KW if k in text.lower())
    score = (pos - neg) / max(pos + neg, 1)
    if pos > neg:
        return score, "Positive"
    elif neg > pos:
        return score, "Negative"
    return 0.0, "Neutral"


def detect_demand_driver(text: str) -> str:
    if pd.isna(text):
        return "Neutral"
    text_l = text.lower()
    boost  = sum(1 for k in DEMAND_BOOST_KW   if k in text_l)
    reduce = sum(1 for k in DEMAND_REDUCE_KW  if k in text_l)
    if boost > reduce:
        return "Boost"
    elif reduce > boost:
        return "Reduce"
    return "Neutral"


def top_ngrams(corpus: list[str], n: int = 2, top_k: int = 20) -> list[tuple]:
    """Return top_k n-grams from corpus using TF-IDF."""
    vectorizer = TfidfVectorizer(
        ngram_range=(n, n), max_features=200,
        stop_words="english", token_pattern=r"[a-zA-Z]{3,}"
    )
    try:
        X = vectorizer.fit_transform(corpus)
        scores = X.mean(axis=0).A1
        vocab  = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:top_k]
        return [(vocab[i], round(scores[i], 4)) for i in top_idx]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────
def run_nlp_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich df with NLP features and print insights."""
    df = df.copy()

    text_cols = ["Event", "Market_News", "Press_Release"]

    # ── Per-column sentiment ──────────────────────────────────────
    for col in text_cols:
        df[[f"{col}_score", f"{col}_sentiment"]] = df[col].apply(
            lambda t: pd.Series(sentiment_score(t)))
        df[f"{col}_driver"] = df[col].apply(detect_demand_driver)

    df["Combined_Text"] = (df["Event"].fillna("") + " " +
                           df["Market_News"].fillna("") + " " +
                           df["Press_Release"].fillna(""))
    df["Avg_Sentiment_Score"] = df[[f"{c}_score" for c in text_cols]].mean(axis=1)

    # Majority sentiment
    def majority(row):
        vals = [row[f"{c}_sentiment"] for c in text_cols]
        return max(set(vals), key=vals.count)
    df["Final_Sentiment"] = df.apply(majority, axis=1)

    def majority_driver(row):
        vals = [row[f"{c}_driver"] for c in text_cols]
        return max(set(vals), key=vals.count)
    df["Demand_Driver"] = df.apply(majority_driver, axis=1)

    return df


def print_drug_sentiment_summary(df: pd.DataFrame):
    print("\n── Sentiment Distribution (Overall) ──────────────")
    print(df["Final_Sentiment"].value_counts().to_string())

    print("\n── Demand Driver Distribution ─────────────────────")
    print(df["Demand_Driver"].value_counts().to_string())

    print("\n── Average Demand by Sentiment ────────────────────")
    print(df.groupby("Final_Sentiment")["Demand"].mean().round(2).to_string())

    print("\n── Average Demand by Demand Driver ────────────────")
    print(df.groupby("Demand_Driver")["Demand"].mean().round(2).to_string())


def extract_top_keywords(df: pd.DataFrame):
    print("\n── Top Bigrams from All Text ───────────────────────")
    corpus = df["Combined_Text"].dropna().tolist()
    bigrams = top_ngrams(corpus, n=2, top_k=15)
    for phrase, score in bigrams:
        print(f"  {phrase:<30} score={score:.4f}")

    print("\n── Top Bigrams by Drug ──────────────────────────────")
    for drug in df["Drug"].unique():
        sub_corpus = df[df["Drug"] == drug]["Combined_Text"].dropna().tolist()
        top = top_ngrams(sub_corpus, n=2, top_k=5)
        if top:
            kws = ", ".join(t[0] for t in top)
            print(f"  {drug:<12}: {kws}")


# ─────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────
def plot_sentiment_demand(df: pd.DataFrame, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#F8FAFC")

    # Sentiment distribution
    counts = df["Final_Sentiment"].value_counts()
    axes[0].bar(counts.index, counts.values, color=PALETTE[:len(counts)], width=0.5)
    axes[0].set_title("Sentiment Distribution", fontsize=13, fontweight="bold")
    axes[0].set_facecolor("#F8FAFC"); axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=10)

    # Avg demand by sentiment
    avg = df.groupby("Final_Sentiment")["Demand"].mean()
    axes[1].bar(avg.index, avg.values, color=PALETTE[:len(avg)], width=0.5)
    axes[1].set_title("Avg Demand by Sentiment", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Avg Demand")
    axes[1].set_facecolor("#F8FAFC"); axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(avg.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_demand_driver(df: pd.DataFrame, fname: str):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8FAFC")
    grp = df.groupby(["Drug", "Demand_Driver"])["Demand"].mean().unstack("Demand_Driver").fillna(0)
    grp.plot(kind="bar", ax=ax, color=PALETTE[:3], width=0.6)
    ax.set_title("Avg Demand by Drug and Demand Driver Signal", fontsize=13, fontweight="bold")
    ax.set_xlabel("Drug"); ax.set_ylabel("Avg Demand")
    ax.set_facecolor("#F8FAFC"); ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Driver", fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, fname: str):
    """Correlation between sentiment score and demand."""
    text_cols = ["Event", "Market_News", "Press_Release"]
    score_cols = [f"{c}_score" for c in text_cols] + ["Avg_Sentiment_Score", "Demand"]
    corr = df[score_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8FAFC")
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    short_labels = [c.replace("_score","").replace("_"," ") for c in corr.columns]
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Sentiment–Demand Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Drug Demand Forecasting — NLP Analysis")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM drug_demand", conn)
    conn.close()

    df = run_nlp_analysis(df)
    print_drug_sentiment_summary(df)
    extract_top_keywords(df)

    print("\n[PLOTS] Generating NLP plots …")
    plot_sentiment_demand(df, f"{OUTPUT_DIR}/nlp_sentiment_demand.png")
    plot_demand_driver(df,    f"{OUTPUT_DIR}/nlp_demand_driver.png")
    plot_correlation_heatmap(df, f"{OUTPUT_DIR}/nlp_correlation.png")

    print(f"\n✓ NLP plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
