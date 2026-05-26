"""
========================================================
  Credit Worthiness Prediction - ML Pipeline
  Algorithms: Logistic Regression, Decision Tree, Random Forest
========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Portable paths (works on Windows, Mac, Linux) ──────────────────────────
BASE_DIR     = Path(__file__).parent          # folder where this script lives
DATA_DIR     = BASE_DIR / "data"
REPORTS_DIR  = BASE_DIR / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


def generate_credit_dataset(n_samples=2000, random_state=42):

    np.random.seed(random_state)
    n = n_samples

    age             = np.random.randint(21, 70, n)
    income          = np.random.normal(55000, 22000, n).clip(15000, 200000)
    loan_amount     = np.random.normal(12000, 8000, n).clip(1000, 60000)
    loan_tenure     = np.random.choice([12, 24, 36, 48, 60], n)
    employment_yrs  = np.random.exponential(5, n).clip(0, 40)
    num_credit_lines= np.random.randint(1, 10, n)
    existing_debts  = np.random.normal(8000, 5000, n).clip(0, 50000)
    missed_payments = np.random.poisson(1.2, n).clip(0, 12)
    credit_score    = np.random.normal(650, 80, n).clip(300, 850)
    education       = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n,
                                        p=[0.30, 0.40, 0.22, 0.08])
    employment_type = np.random.choice(["Salaried", "Self-Employed", "Business", "Unemployed"], n,
                                        p=[0.55, 0.20, 0.18, 0.07])


    debt_to_income     = existing_debts / (income + 1)
    loan_to_income     = loan_amount / (income + 1)
    payment_history_score = 100 - (missed_payments * 8)

    # Target: creditworthy (1=good, 0=bad)
    score = (
          0.30 * (credit_score / 850)
        + 0.20 * (income / 200000)
        + 0.20 * (1 - debt_to_income.clip(0, 1))
        + 0.15 * (payment_history_score / 100)
        + 0.10 * (employment_yrs / 40)
        + 0.05 * np.random.rand(n)       # noise
    )
    creditworthy = (score > 0.52).astype(int)

    df = pd.DataFrame({
        "age":                  age,
        "income":               income.round(2),
        "loan_amount":          loan_amount.round(2),
        "loan_tenure_months":   loan_tenure,
        "employment_years":     employment_yrs.round(1),
        "num_credit_lines":     num_credit_lines,
        "existing_debts":       existing_debts.round(2),
        "missed_payments":      missed_payments,
        "credit_score":         credit_score.round(0).astype(int),
        "debt_to_income_ratio": debt_to_income.round(4),
        "loan_to_income_ratio": loan_to_income.round(4),
        "payment_history_score":payment_history_score.clip(0, 100).round(1),
        "education":            education,
        "employment_type":      employment_type,
        "creditworthy":         creditworthy,
    })
    return df


def preprocess(df):
    df = df.copy()


    le_edu = LabelEncoder()
    le_emp = LabelEncoder()
    df["education_enc"]       = le_edu.fit_transform(df["education"])
    df["employment_type_enc"] = le_emp.fit_transform(df["employment_type"])


    df["income_per_year_employed"] = df["income"] / (df["employment_years"] + 1)
    df["monthly_loan_burden"]      = df["loan_amount"] / df["loan_tenure_months"]
    df["credit_utilization"]       = df["existing_debts"] / (df["num_credit_lines"] * 5000 + 1)

    feature_cols = [
        "age", "income", "loan_amount", "loan_tenure_months",
        "employment_years", "num_credit_lines", "existing_debts",
        "missed_payments", "credit_score", "debt_to_income_ratio",
        "loan_to_income_ratio", "payment_history_score",
        "education_enc", "employment_type_enc",
        "income_per_year_employed", "monthly_loan_burden", "credit_utilization",
    ]

    X = df[feature_cols]
    y = df["creditworthy"]
    return X, y, feature_cols


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }
    return metrics, model, y_pred, y_prob


def plot_all(results_df, models_info, X_test, y_test, feature_cols, df):
    sns.set_theme(style="darkgrid", palette="muted")
    colors = {"Logistic Regression": "#4E9AF1",
              "Decision Tree":       "#F17C4E",
              "Random Forest":       "#4EBF8A"}


    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    for ax, metric in zip(axes, metrics_cols):
        bars = ax.bar(results_df["Model"], results_df[metric],
                      color=[colors[m] for m in results_df["Model"]],
                      edgecolor="white", linewidth=0.8, width=0.5)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0.7, 1.0)
        ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "01_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    for name, model, y_prob in models_info:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC={auc_val:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "02_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, (name, model, y_prob) in zip(axes, models_info):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Credit", "Creditworthy"],
                    yticklabels=["Not Credit", "Creditworthy"],
                    linewidths=0.5, linecolor="white")
        ax.set_title(name, fontsize=11, fontweight="bold", color=colors[name])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "03_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    rf_model = next(m for n, m, _ in models_info if n == "Random Forest")
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(importances.index, importances.values, color="#4EBF8A", edgecolor="white")
    ax.set_title("Feature Importances – Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "04_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    sns.heatmap(numeric_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax, linewidths=0.4,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅  All 5 plots saved to reports/")



def main():
    print("=" * 60)
    print("   CREDIT WORTHINESS PREDICTION – ML PIPELINE")
    print("=" * 60)


    print("\n📊  Generating dataset …")
    df = generate_credit_dataset(n_samples=2000)
    df.to_csv(DATA_DIR / "credit_data.csv", index=False)
    print(f"    Samples: {len(df):,}  |  Creditworthy: {df['creditworthy'].mean()*100:.1f}%")


    X, y, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)


    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }

    results     = []
    models_info = []      # (name, fitted_model, y_prob)

    print("\n🚀  Training models …\n")
    for name, model in models.items():
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc  if name == "Logistic Regression" else X_test
        metrics, fitted, y_pred, y_prob = evaluate_model(name, model, X_tr, X_te, y_train, y_test)
        results.append(metrics)
        models_info.append((name, fitted, y_prob))
        print(f"  {'─'*40}")
        print(f"  {name}")
        for k, v in metrics.items():
            if k != "Model":
                print(f"    {k:<12}: {v:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    best = results_df.loc[results_df["ROC-AUC"].idxmax(), "Model"]
    print(f"\n🏆  Best model by ROC-AUC: {best}")


    print(f"\n🔁  5-Fold CV for {best} …")
    best_model = next(m for n, m, _ in models_info if n == best)
    X_full = X_test_sc if best == "Logistic Regression" else X_test
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_full, y_test, cv=cv, scoring="roc_auc")
    print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


    print("\n📈  Generating visualisation plots …")
    plot_all(results_df, models_info, X_test, y_test, feature_cols, df)

    print("\n✅  Pipeline complete!")
    print("   Outputs → data/credit_data.csv")
    print("             reports/model_metrics.csv")
    print("             reports/01_metrics_comparison.png … 05_correlation_heatmap.png")


if __name__ == "__main__":
    main()