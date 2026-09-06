"""
Heart Disease Prediction — End-to-End Data Science Project
Dataset: UCI Heart Disease (Cleveland) — 303 patients, 13 clinical features
Goal: Predict presence of heart disease (target: 1 = disease, 0 = no disease)

Pipeline: Load -> Clean -> EDA -> Feature Engineering -> Modeling -> Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
print(df.head())

report = {}
report["shape_raw"] = df.shape

# ----------------------------------------------------------------------
# 2. INITIAL INSPECTION
# ----------------------------------------------------------------------
report["missing_values"] = df.isnull().sum().to_dict()
report["duplicates"] = int(df.duplicated().sum())
report["dtypes"] = df.dtypes.astype(str).to_dict()

# ----------------------------------------------------------------------
# 3. CLEANING
# ----------------------------------------------------------------------
df = df.drop_duplicates()
report["shape_after_dedup"] = df.shape

# column meanings (documented for readability / interview talking points)
col_meaning = {
    "age": "Age in years",
    "sex": "1 = male, 0 = female",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (1=normal,2=fixed defect,3=reversible defect)",
    "target": "1 = heart disease present, 0 = no heart disease",
}

# ----------------------------------------------------------------------
# 4. EDA
# ----------------------------------------------------------------------

# 4a. Target distribution
plt.figure()
sns.countplot(x="target", data=df, palette="Set2")
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.savefig(f"{OUT}/01_target_distribution.png", bbox_inches="tight", dpi=120)
plt.close()

# 4b. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{OUT}/02_correlation_heatmap.png", bbox_inches="tight", dpi=120)
plt.close()

# 4c. Age distribution by target
plt.figure()
sns.histplot(data=df, x="age", hue="target", kde=True, bins=20, palette="Set1")
plt.title("Age Distribution by Heart Disease Status")
plt.savefig(f"{OUT}/03_age_distribution.png", bbox_inches="tight", dpi=120)
plt.close()

# 4d. Chest pain type vs target
plt.figure()
sns.countplot(x="cp", hue="target", data=df, palette="Set2")
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type (0-3)")
plt.savefig(f"{OUT}/04_chestpain_vs_target.png", bbox_inches="tight", dpi=120)
plt.close()

# 4e. Max heart rate vs age, colored by target
plt.figure()
sns.scatterplot(data=df, x="age", y="thalach", hue="target", palette="Set1")
plt.title("Max Heart Rate vs Age")
plt.savefig(f"{OUT}/05_heartrate_vs_age.png", bbox_inches="tight", dpi=120)
plt.close()

report["target_balance"] = df["target"].value_counts(normalize=True).to_dict()

# ----------------------------------------------------------------------
# 5. FEATURE ENGINEERING + TRAIN/TEST SPLIT
# ----------------------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 6. MODELING — compare 3 algorithms
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": SVC(probability=True, random_state=42),
}

results = {}
roc_data = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
    }

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, auc)

    # confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT}/06_confusion_matrix_{safe_name}.png", bbox_inches="tight", dpi=120)
    plt.close()

report["model_results"] = results

# 6b. ROC curve comparison plot
plt.figure()
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig(f"{OUT}/07_roc_comparison.png", bbox_inches="tight", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (from Random Forest — best interpretability)
# ----------------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
report["feature_importance"] = importances.round(4).to_dict()

plt.figure()
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.savefig(f"{OUT}/08_feature_importance.png", bbox_inches="tight", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 8. SAVE REPORT
# ----------------------------------------------------------------------
best_model = max(results, key=lambda k: results[k]["roc_auc"])
report["best_model"] = best_model

with open(f"{OUT}/metrics_report.json", "w") as f:
    json.dump(report, f, indent=2)

# ----------------------------------------------------------------------
# 9. SAVE MODEL + SCALER (for the Streamlit app)
# ----------------------------------------------------------------------
import joblib
joblib.dump(models[best_model], "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print(f"\nSaved model.pkl ({best_model}), scaler.pkl, feature_names.pkl for deployment.")

print("\n=== SUMMARY ===")
for name, r in results.items():
    print(name, r)
print("Best model (by ROC-AUC):", best_model)
print("\nAll plots and metrics saved to:", OUT)
