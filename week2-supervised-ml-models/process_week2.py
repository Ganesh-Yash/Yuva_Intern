"""
Week 2 Task - Supervised Machine Learning Models
Builds a fresh sample HR dataset (same schema and cleaning pipeline as
Week 1), this time with genuine relationships baked into the target
variables so the models trained below have real signal to learn from,
then trains and compares supervised models on a classification task
(predict Attrition) and a regression task (predict Monthly Salary).
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

ASSETS = "/home/claude/week2/week2_assets"
os.makedirs(ASSETS, exist_ok=True)
stats = {}

# ---------------------------------------------------------------------------
# STEP 0: Sample dataset. Same schema/pipeline as Week 1, but this time
# Salary and Attrition are generated FROM the other features (with noise)
# so the supervised models below have real relationships to learn.
# ---------------------------------------------------------------------------
np.random.seed(42)
n = 600

departments = ["Sales", "IT", "HR", "Finance", "Marketing"]
cities = ["New York", "Chicago", "Houston", "Phoenix", "Los Angeles"]
genders = ["Male", "Female"]
dept_salary_boost = {"IT": 8000, "Finance": 6000, "Sales": 2000, "Marketing": 0, "HR": -2000}

age = np.random.randint(21, 60, n)
years_exp = np.clip((age - 21) * np.random.uniform(0.25, 0.85, n), 0, 35).round(0)
department = np.random.choice(departments, n)
gender = np.random.choice(genders, n)
city = np.random.choice(cities, n)
performance = np.random.randint(1, 6, n)

salary = (32000
          + years_exp * 950
          + performance * 2200
          + np.array([dept_salary_boost[d] for d in department])
          + np.random.normal(0, 4000, n))
salary = np.clip(salary.round(2), 20000, None)

logit = (-1.3 - 0.65 * (performance - 3) - 0.09 * (years_exp - 8) + np.random.normal(0, 0.65, n))
prob = 1 / (1 + np.exp(-logit))
attrition = np.where(np.random.rand(n) < prob, "Yes", "No")

df = pd.DataFrame({
    "EmployeeID": range(2001, 2001 + n),
    "Age": age.astype(float),
    "Gender": gender,
    "Department": department,
    "City": city,
    "YearsExperience": years_exp.astype(float),
    "MonthlySalary": salary,
    "PerformanceScore": performance,
    "Attrition": attrition,
})

# same messiness as Week 1: missing values, inconsistent text, duplicates
for col in ["Age", "MonthlySalary", "YearsExperience", "Department"]:
    idx = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    df.loc[idx, col] = np.nan
inconsistent_idx = np.random.choice(df.index, size=int(n * 0.055), replace=False)
df.loc[inconsistent_idx, "City"] = df.loc[inconsistent_idx, "City"].str.lower()
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

raw_shape = df.shape
df.to_csv("/home/claude/week2/week2_raw_employee_data.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 1: Cleaning + imputation + outlier capping + encoding
# (condensed version of the Week 1 pipeline -- see Week 1 report for detail)
# ---------------------------------------------------------------------------
dup_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
df["City"] = df["City"].str.strip().str.title()

df["Age"] = df["Age"].fillna(df["Age"].median())
df["YearsExperience"] = df["YearsExperience"].fillna(df["YearsExperience"].median())
df["MonthlySalary"] = df["MonthlySalary"].fillna(df["MonthlySalary"].median())
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

Q1, Q3 = df["MonthlySalary"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
n_outliers = int(((df["MonthlySalary"] < lower) | (df["MonthlySalary"] > upper)).sum())
df["MonthlySalary"] = df["MonthlySalary"].clip(lower=lower, upper=upper)

df = df.drop(columns=["EmployeeID"])

le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
le_attr = LabelEncoder()
df["Attrition"] = le_attr.fit_transform(df["Attrition"])  # No=0, Yes=1
df = pd.get_dummies(df, columns=["Department", "City"], prefix=["Dept", "City"], drop_first=True, dtype=int)

clean_shape = df.shape
df.to_csv("/home/claude/week2/week2_model_ready_unscaled.csv", index=False)

stats["raw_rows"] = int(raw_shape[0])
stats["dup_count"] = dup_count
stats["clean_rows"] = int(clean_shape[0])
stats["clean_cols"] = int(clean_shape[1])
stats["n_outliers"] = n_outliers
stats["attrition_mapping"] = {k: int(v) for k, v in zip(le_attr.classes_, le_attr.transform(le_attr.classes_))}

vc = df["Attrition"].value_counts()
stats["attrition_counts"] = {("Yes" if k == 1 else "No"): int(v) for k, v in vc.items()}
tot = int(vc.sum())
stats["attrition_pct"] = {k: round(v / tot * 100, 1) for k, v in stats["attrition_counts"].items()}
stats["baseline_accuracy"] = round(max(stats["attrition_counts"].values()) / tot * 100, 1)

# ---------------------------------------------------------------------------
# TASK A: CLASSIFICATION -- predict Attrition
# ---------------------------------------------------------------------------
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scale_cols = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore"]
scaler = MinMaxScaler()
X_train, X_test = X_train.copy(), X_test.copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

stats["clf_train_size"] = int(len(X_train))
stats["clf_test_size"] = int(len(X_test))

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

clf_results = {}
roc_data = {}
for name, model in clf_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr.tolist(), tpr.tolist())
    clf_results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        "confusion_matrix": cm.tolist(),
    }

stats["clf_results"] = clf_results
best_clf = max(clf_results, key=lambda k: clf_results[k]["f1"])
stats["best_clf"] = best_clf

# confusion matrices grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, (name, res) in zip(axes.flat, clf_results.items()):
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{ASSETS}/confusion_matrices.png")
plt.close()

# ROC curves
plt.figure(figsize=(6.5, 5.5))
for name, (fpr, tpr) in roc_data.items():
    auc_val = clf_results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves \u2014 Attrition Classifiers")
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(f"{ASSETS}/roc_curves.png")
plt.close()

# metric comparison bar chart
metric_df = pd.DataFrame(clf_results).T[["accuracy", "precision", "recall", "f1"]].astype(float)
metric_df.plot(kind="bar", figsize=(7.2, 4.4), colormap="viridis")
plt.title("Classification Metric Comparison")
plt.ylabel("Score (%)")
plt.xticks(rotation=15)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{ASSETS}/clf_metric_comparison.png")
plt.close()

# feature importance from random forest
rf = clf_models["Random Forest"]
importances_desc = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(8)
plt.figure(figsize=(6.5, 4.4))
importances_desc.sort_values().plot(kind="barh", color="#2E75B6")
plt.title("Top Feature Importances \u2014 Random Forest (Attrition)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_importance.png")
plt.close()
stats["top_features_clf"] = importances_desc.round(3).to_dict()

# ---------------------------------------------------------------------------
# TASK B: REGRESSION -- predict MonthlySalary
# ---------------------------------------------------------------------------
Xr = df.drop(columns=["MonthlySalary"])
yr = df["MonthlySalary"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

scale_cols_r = ["Age", "YearsExperience", "PerformanceScore"]
scaler_r = MinMaxScaler()
Xr_train, Xr_test = Xr_train.copy(), Xr_test.copy()
Xr_train[scale_cols_r] = scaler_r.fit_transform(Xr_train[scale_cols_r])
Xr_test[scale_cols_r] = scaler_r.transform(Xr_test[scale_cols_r])

stats["reg_train_size"] = int(len(Xr_train))
stats["reg_test_size"] = int(len(Xr_test))

reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
}

reg_results = {}
reg_preds = {}
for name, model in reg_models.items():
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)
    reg_preds[name] = pred
    reg_results[name] = {
        "mae": round(float(mean_absolute_error(yr_test, pred)), 2),
        "mse": round(float(mean_squared_error(yr_test, pred)), 2),
        "rmse": round(float(mean_squared_error(yr_test, pred) ** 0.5), 2),
        "r2": round(float(r2_score(yr_test, pred)), 3),
    }
stats["reg_results"] = reg_results
best_reg = max(reg_results, key=lambda k: reg_results[k]["r2"])
stats["best_reg"] = best_reg

# predicted vs actual scatter
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (name, pred) in zip(axes, reg_preds.items()):
    ax.scatter(yr_test, pred, alpha=0.5, s=18, color="#8E44AD")
    lims = [min(yr_test.min(), pred.min()), max(yr_test.max(), pred.max())]
    ax.plot(lims, lims, "--", color="gray")
    ax.set_xlabel("Actual Monthly Salary")
    ax.set_ylabel("Predicted Monthly Salary")
    ax.set_title(f"{name}\nR\u00b2 = {reg_results[name]['r2']}")
plt.tight_layout()
plt.savefig(f"{ASSETS}/reg_pred_vs_actual.png")
plt.close()

# linear regression coefficients
lr = reg_models["Linear Regression"]
coef = pd.Series(lr.coef_, index=Xr_train.columns).sort_values()
stats["lr_coefficients_top"] = coef.round(1).tail(6).to_dict()
stats["lr_coefficients_bottom"] = coef.round(1).head(4).to_dict()
stats["lr_intercept"] = round(float(lr.intercept_), 2)

plt.figure(figsize=(6.5, 5))
colors = ["#C0392B" if v < 0 else "#27AE60" for v in coef]
coef.plot(kind="barh", color=colors)
plt.title("Linear Regression Coefficients (Monthly Salary)")
plt.xlabel("Coefficient (\u20b9 per unit / per category)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/lr_coefficients.png")
plt.close()

with open(f"{ASSETS}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("DONE.")
print("raw_shape", raw_shape, "clean_shape", clean_shape)
print("attrition_counts", stats["attrition_counts"], "baseline_acc", stats["baseline_accuracy"])
print("best_clf", best_clf, clf_results[best_clf])
print("best_reg", best_reg, reg_results[best_reg])
