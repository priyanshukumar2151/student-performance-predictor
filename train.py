"""
Student Performance Predictor - training script
Dataset: UCI "Student Performance" (student-mat.csv, semicolon separated)
Download: https://archive.ics.uci.edu/dataset/320/student+performance
Put student-mat.csv in the same folder as this file, then run:  python train.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# 1. Load data -------------------------------------------------------------
df = pd.read_csv("student-mat.csv", sep=";")
print("Shape:", df.shape)
print(df.head())
print("Missing values:", df.isnull().sum().sum())

# 2. Target: pass (G3 >= 10) / fail ------------------------------------------
df["pass"] = (df["G3"] >= 10).astype(int)

# We predict WITHOUT G1/G2/G3 so the model uses habits, not previous grades
features = ["studytime", "failures", "absences", "age", "Medu", "Fedu",
            "traveltime", "freetime", "goout", "Dalc", "Walc", "health",
            "famrel", "schoolsup", "famsup", "paid", "activities",
            "internet", "higher", "romantic"]
X = df[features].copy()
y = df["pass"]

# Encode yes/no columns
for col in ["schoolsup", "famsup", "paid", "activities", "internet", "higher", "romantic"]:
    X[col] = (X[col] == "yes").astype(int)

# 3. EDA charts ---------------------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(5, 4)); y.value_counts().plot(kind="bar", color=["#e74c3c", "#2ecc71"])
plt.xticks([0, 1], ["Fail", "Pass"], rotation=0); plt.title("Pass / Fail count")
plt.tight_layout(); plt.savefig("eda_class_balance.png", dpi=150); plt.close()

plt.figure(figsize=(9, 7)); sns.heatmap(pd.concat([X, y], axis=1).corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig("eda_correlation.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4)); sns.boxplot(x=y, y=X["absences"]); plt.xticks([0, 1], ["Fail", "Pass"])
plt.title("Absences vs result"); plt.tight_layout(); plt.savefig("eda_absences.png", dpi=150); plt.close()

# 4. Train / test split ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([("num", StandardScaler(), list(X.columns))])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# 5. Compare models ----------------------------------------------------------------
rows = []
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred)})
results = pd.DataFrame(rows).round(3)
print("\n=== Model comparison ===\n", results.to_string(index=False))
results.to_csv("model_comparison.csv", index=False)

# 6. Tune Random Forest ------------------------------------------------------------
pipe = Pipeline([("pre", pre), ("model", RandomForestClassifier(random_state=42))])
grid = GridSearchCV(pipe, {"model__n_estimators": [100, 200, 300],
                           "model__max_depth": [None, 5, 10],
                           "model__min_samples_split": [2, 5]},
                    cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)
print("\nBest params:", grid.best_params_)
print("Tuned accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["Fail", "Pass"]))

# 7. Confusion matrix + feature importance --------------------------------------------
plt.figure(figsize=(4.5, 4))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (tuned RF)")
plt.tight_layout(); plt.savefig("result_confusion_matrix.png", dpi=150); plt.close()

imp = pd.Series(best.named_steps["model"].feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 6)); imp.tail(12).plot(kind="barh", color="#3498db")
plt.title("Top feature importances"); plt.tight_layout()
plt.savefig("result_feature_importance.png", dpi=150); plt.close()

# 8. Save model -------------------------------------------------------------------------
joblib.dump({"model": best, "features": list(X.columns)}, "model.joblib")
print("\nSaved model.joblib and charts (PNG files). Take screenshots of these for submission.")
