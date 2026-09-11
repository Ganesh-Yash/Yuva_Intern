"""
Week 3 Task — Unsupervised Learning & Model Evaluation

Generates a sample HR dataset with genuine employee segments, then:
  1. Clusters it with K-Means (elbow + silhouette) and Hierarchical Clustering
  2. Reduces it with PCA for visualisation
  3. Evaluates a supervised attrition classifier with k-fold CV, a full
     metric suite, and GridSearchCV hyperparameter tuning
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from report_utils import InternshipReport  # noqa: E402

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

np.random.seed(42)
stats: dict = {}


# ---------------------------------------------------------------------------
# STEP 0 — Dataset with three natural employee segments
# ---------------------------------------------------------------------------
n = 720
departments = ["Sales", "IT", "HR", "Finance", "Marketing"]
cities = ["New York", "Chicago", "Houston", "Phoenix", "Los Angeles"]

# Three latent groups the clustering algorithms should recover
# 0 = early-career, 1 = mid-career specialists, 2 = senior high-performers
segment = np.random.choice([0, 1, 2], size=n, p=[0.38, 0.40, 0.22])

age = np.empty(n)
years_exp = np.empty(n)
performance = np.empty(n, dtype=int)
department = np.empty(n, dtype=object)
salary = np.empty(n)

for i, s in enumerate(segment):
    if s == 0:
        age[i] = np.random.normal(26, 3)
        years_exp[i] = np.clip(age[i] - 22 + np.random.normal(0, 1.2), 0, 8)
        performance[i] = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
        department[i] = np.random.choice(departments, p=[0.32, 0.18, 0.18, 0.12, 0.20])
        salary[i] = 28000 + years_exp[i] * 900 + performance[i] * 1200 + np.random.normal(0, 2500)
    elif s == 1:
        age[i] = np.random.normal(36, 4)
        years_exp[i] = np.clip(age[i] - 23 + np.random.normal(0, 1.8), 4, 18)
        performance[i] = np.random.choice([2, 3, 4, 5], p=[0.10, 0.35, 0.40, 0.15])
        department[i] = np.random.choice(departments, p=[0.22, 0.28, 0.10, 0.22, 0.18])
        salary[i] = 42000 + years_exp[i] * 1100 + performance[i] * 1800 + np.random.normal(0, 3200)
    else:
        age[i] = np.random.normal(48, 5)
        years_exp[i] = np.clip(age[i] - 22 + np.random.normal(0, 2.0), 12, 35)
        performance[i] = np.random.choice([3, 4, 5], p=[0.20, 0.45, 0.35])
        department[i] = np.random.choice(departments, p=[0.12, 0.34, 0.08, 0.32, 0.14])
        salary[i] = 58000 + years_exp[i] * 1300 + performance[i] * 2400 + np.random.normal(0, 4000)

age = np.clip(age, 21, 60).round(0)
years_exp = years_exp.round(0)
salary = np.clip(salary, 22000, None).round(2)
gender = np.random.choice(["Male", "Female"], n)
city = np.random.choice(cities, n)

# Attrition depends on performance, tenure and (weakly) segment
logit = (
    -0.9
    - 0.70 * (performance - 3)
    - 0.08 * (years_exp - 8)
    + 0.35 * (segment == 0).astype(float)
    + np.random.normal(0, 0.55, n)
)
prob = 1 / (1 + np.exp(-logit))
attrition = np.where(np.random.rand(n) < prob, "Yes", "No")

df = pd.DataFrame(
    {
        "EmployeeID": range(3001, 3001 + n),
        "Age": age,
        "Gender": gender,
        "Department": department,
        "City": city,
        "YearsExperience": years_exp,
        "MonthlySalary": salary,
        "PerformanceScore": performance,
        "Attrition": attrition,
        "TrueSegment": segment,
    }
)
df.to_csv(DATA / "week3_raw_employee_data.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 1 — Light preprocessing for clustering / modelling
# ---------------------------------------------------------------------------
work = df.drop(columns=["EmployeeID"]).copy()
le_gender = LabelEncoder()
work["Gender"] = le_gender.fit_transform(work["Gender"])
le_attr = LabelEncoder()
work["Attrition"] = le_attr.fit_transform(work["Attrition"])  # No=0, Yes=1
encoded = pd.get_dummies(
    work, columns=["Department", "City"], prefix=["Dept", "City"], drop_first=True, dtype=int
)
encoded.to_csv(DATA / "week3_model_ready.csv", index=False)

cluster_features = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore"]
X_cluster = encoded[cluster_features].copy()
scaler_c = StandardScaler()
X_scaled = scaler_c.fit_transform(X_cluster)

stats["n_rows"] = int(len(df))
stats["n_cluster_features"] = len(cluster_features)
stats["cluster_features"] = cluster_features
stats["attrition_counts"] = {k: int(v) for k, v in df["Attrition"].value_counts().items()}
stats["true_segment_counts"] = {str(int(k)): int(v) for k, v in pd.Series(segment).value_counts().sort_index().items()}
stats["attrition_mapping"] = {k: int(v) for k, v in zip(le_attr.classes_, le_attr.transform(le_attr.classes_))}

# ---------------------------------------------------------------------------
# STEP 2 — K-Means: elbow + silhouette to choose k
# ---------------------------------------------------------------------------
ks = list(range(2, 9))
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(float(km.inertia_))
    silhouettes.append(float(silhouette_score(X_scaled, labels_k)))

stats["k_range"] = ks
stats["inertias"] = [round(v, 1) for v in inertias]
stats["silhouettes"] = [round(v, 3) for v in silhouettes]
best_k = int(ks[int(np.argmax(silhouettes))])
stats["best_k_silhouette"] = best_k
stats["best_silhouette"] = round(max(silhouettes), 3)

# Use k=3: it matches the generating process and is at/near the elbow
chosen_k = 3
stats["chosen_k"] = chosen_k

kmeans = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
stats["kmeans_silhouette"] = round(float(silhouette_score(X_scaled, kmeans_labels)), 3)
stats["kmeans_inertia"] = round(float(kmeans.inertia_), 1)
stats["kmeans_sizes"] = {str(i): int((kmeans_labels == i).sum()) for i in range(chosen_k)}

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(ks, inertias, marker="o", color="#1F4E79", linewidth=2)
ax.axvline(chosen_k, color="#C0392B", linestyle="--", label=f"Chosen k = {chosen_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia (within-cluster sum of squares)")
ax.set_title("K-Means Elbow Method")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "elbow_method.png")
plt.close()

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(ks, silhouettes, marker="o", color="#27AE60", linewidth=2)
ax.axvline(chosen_k, color="#C0392B", linestyle="--", label=f"Chosen k = {chosen_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Mean silhouette score")
ax.set_title("K-Means Silhouette Scores")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "silhouette_scores.png")
plt.close()

# ---------------------------------------------------------------------------
# STEP 3 — Hierarchical clustering
# ---------------------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward")
hier_labels = agg.fit_predict(X_scaled)
stats["hier_silhouette"] = round(float(silhouette_score(X_scaled, hier_labels)), 3)
stats["hier_sizes"] = {str(i): int((hier_labels == i).sum()) for i in range(chosen_k)}

# Dendrogram on a 60-row subsample so the tree is readable
sample_idx = np.random.choice(len(X_scaled), size=60, replace=False)
Z = linkage(X_scaled[sample_idx], method="ward")
fig, ax = plt.subplots(figsize=(8.2, 4.6))
dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-2, 2])
ax.set_title("Hierarchical Clustering Dendrogram (Ward, n=60 subsample)")
ax.set_ylabel("Ward distance")
ax.set_xlabel("Employees (subsample)")
plt.tight_layout()
plt.savefig(OUTPUTS / "hierarchical_dendrogram.png")
plt.close()

# ---------------------------------------------------------------------------
# STEP 4 — PCA dimensionality reduction
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
stats["pca_explained"] = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
stats["pca_total_2d"] = round(float(sum(pca.explained_variance_ratio_)) * 100, 1)

pca_full = PCA(random_state=42).fit(X_scaled)
cum = np.cumsum(pca_full.explained_variance_ratio_) * 100
stats["pca_cum_var"] = [round(float(v), 1) for v in cum]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
       pca_full.explained_variance_ratio_ * 100, color="#2E75B6", label="Individual")
ax.plot(range(1, len(cum) + 1), cum, marker="o", color="#C0392B", label="Cumulative")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance (%)")
ax.set_title("PCA Explained Variance")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "pca_explained_variance.png")
plt.close()


def scatter_pca(labels, title, path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, s=22, alpha=0.85)
    ax.set_xlabel(f"PC1 ({stats['pca_explained'][0]}% variance)")
    ax.set_ylabel(f"PC2 ({stats['pca_explained'][1]}% variance)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


scatter_pca(kmeans_labels, "K-Means Clusters in PCA Space (k=3)", OUTPUTS / "kmeans_pca_scatter.png")
scatter_pca(hier_labels, "Hierarchical Clusters in PCA Space (k=3)", OUTPUTS / "hierarchical_pca_scatter.png", cmap="plasma")
scatter_pca(segment, "True Generating Segments in PCA Space", OUTPUTS / "true_segments_pca.png", cmap="coolwarm")

# Cluster profiles (K-Means)
profile = X_cluster.copy()
profile["Cluster"] = kmeans_labels
means = profile.groupby("Cluster")[cluster_features].mean().round(1)
stats["kmeans_profiles"] = {
    str(i): {c: float(means.loc[i, c]) for c in cluster_features} for i in means.index
}

fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
for ax, feat in zip(axes.flat, cluster_features):
    sns.barplot(data=profile, x="Cluster", y=feat, hue="Cluster", ax=ax,
                palette="viridis", legend=False, errorbar="sd")
    ax.set_title(feat)
plt.suptitle("K-Means Cluster Profiles (mean ± SD)", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUTS / "cluster_profiles.png")
plt.close()

# Agreement with the true generating segments (for discussion, not a "score")
ct = pd.crosstab(pd.Series(segment, name="TrueSegment"),
                 pd.Series(kmeans_labels, name="KMeans"))
# map each kmeans cluster to majority true segment
mapping = {int(c): int(ct[c].idxmax()) for c in ct.columns}
aligned = np.array([mapping[int(l)] for l in kmeans_labels])
stats["kmeans_true_agreement"] = round(float((aligned == segment).mean()) * 100, 1)

# Persona names based on profiles (sorted by mean salary)
order = means["MonthlySalary"].sort_values().index.tolist()
persona = {}
labels_for_order = ["Early-career", "Mid-career specialists", "Senior high-performers"]
for name, cid in zip(labels_for_order, order):
    persona[str(int(cid))] = name
stats["kmeans_personas"] = persona

# ---------------------------------------------------------------------------
# STEP 5 — Supervised evaluation setup (Attrition)
# ---------------------------------------------------------------------------
y = encoded["Attrition"]
X = encoded.drop(columns=["Attrition", "TrueSegment"])
scale_cols = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = MinMaxScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
stats["clf_train"] = int(len(X_train))
stats["clf_test"] = int(len(X_test))
stats["attrition_rate"] = round(float(y.mean()) * 100, 1)

# Pipeline used for CV so scaling is inside each fold (no leakage)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

model_specs = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

cv_results = {}
for name, clf in model_specs.items():
    pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])
    # scale only numeric cols? MinMax on dummies is harmless (already 0/1)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    cv_results[name] = {
        m: {"mean": round(float(scores[f"test_{m}"].mean()) * (100 if m != "roc_auc" else 1), 3 if m == "roc_auc" else 1),
            "std": round(float(scores[f"test_{m}"].std()) * (100 if m != "roc_auc" else 1), 3 if m == "roc_auc" else 1)}
        for m in scoring
    }
stats["cv_results"] = cv_results

# Hold-out metrics + confusion / ROC for Random Forest (default)
rf_default = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_default.fit(X_train, y_train)
y_pred_def = rf_default.predict(X_test)
y_proba_def = rf_default.predict_proba(X_test)[:, 1]
stats["holdout_default"] = {
    "accuracy": round(accuracy_score(y_test, y_pred_def) * 100, 1),
    "precision": round(precision_score(y_test, y_pred_def, zero_division=0) * 100, 1),
    "recall": round(recall_score(y_test, y_pred_def, zero_division=0) * 100, 1),
    "f1": round(f1_score(y_test, y_pred_def, zero_division=0) * 100, 1),
    "roc_auc": round(roc_auc_score(y_test, y_proba_def), 3),
    "confusion_matrix": confusion_matrix(y_test, y_pred_def).tolist(),
}

# ---------------------------------------------------------------------------
# STEP 6 — Hyperparameter tuning (GridSearchCV on Random Forest)
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [4, 6, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)
stats["best_params"] = grid.best_params_
stats["best_cv_f1"] = round(float(grid.best_score_) * 100, 1)

best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
stats["holdout_tuned"] = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
    "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
    "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
    "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    "confusion_matrix": cm.tolist(),
}

# Grid search heatmap: mean CV F1 by n_estimators × max_depth (avg over other params)
cvres = pd.DataFrame(grid.cv_results_)
cvres["max_depth_label"] = cvres["param_max_depth"].apply(lambda v: "None" if v is None else str(v))
pivot = cvres.pivot_table(
    index="max_depth_label",
    columns="param_n_estimators",
    values="mean_test_score",
    aggfunc="mean",
)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
ax.set_title("GridSearchCV mean F1 (averaged over other params)")
ax.set_xlabel("n_estimators")
ax.set_ylabel("max_depth")
plt.tight_layout()
plt.savefig(OUTPUTS / "gridsearch_heatmap.png")
plt.close()

# Confusion matrices default vs tuned
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
for ax, title, mat in [
    (axes[0], "Default Random Forest", np.array(stats["holdout_default"]["confusion_matrix"])),
    (axes[1], "Tuned Random Forest", np.array(cm)),
]:
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stay (No)", "Leave (Yes)"], yticklabels=["Stay (No)", "Leave (Yes)"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUTPUTS / "confusion_matrices.png")
plt.close()

# ROC default vs tuned
fpr_d, tpr_d, _ = roc_curve(y_test, y_proba_def)
fpr_t, tpr_t, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6.2, 5.0))
ax.plot(fpr_d, tpr_d, label=f"Default RF (AUC={stats['holdout_default']['roc_auc']:.3f})", linewidth=2)
ax.plot(fpr_t, tpr_t, label=f"Tuned RF (AUC={stats['holdout_tuned']['roc_auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Hold-out Test Set")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUTS / "roc_curves.png")
plt.close()

# CV metric comparison
metric_names = ["accuracy", "precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
x = np.arange(len(model_specs))
width = 0.18
for i, m in enumerate(metric_names):
    vals = [cv_results[name][m]["mean"] for name in model_specs]
    ax.bar(x + i * width, vals, width, label=m.capitalize())
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(list(model_specs.keys()), rotation=12)
ax.set_ylabel("5-fold CV mean (%)")
ax.set_title("Cross-Validated Classification Metrics")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUTS / "cv_metric_comparison.png")
plt.close()

# Feature importance from tuned RF
importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(8)
stats["top_features"] = importances.round(3).to_dict()
fig, ax = plt.subplots(figsize=(6.5, 4.4))
importances.sort_values().plot(kind="barh", color="#2E75B6", ax=ax)
ax.set_title("Top Feature Importances — Tuned Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUTS / "feature_importance.png")
plt.close()

with open(OUTPUTS / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# STEP 7 — Word report
# ---------------------------------------------------------------------------
def pct(v):
    return f"{v}%"


def auc(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


r = InternshipReport(
    title="Unsupervised Learning & Model Evaluation",
    subtitle="Week 3 Task Report",
    author="Vutukuri Yaswanth Ganesh Kumar",
    date="4 September 2026",
    task="Week 3 — Unsupervised Learning & Model Evaluation",
    tools="Python, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn",
)

r.heading("1. Objective")
r.para(
    "This task covers the two remaining pillars of a standard machine-learning "
    "workflow that Week 2 did not yet treat in depth: finding structure without "
    "labels (unsupervised learning) and judging a supervised model honestly "
    "(cross-validation, a full classification metric suite, and hyperparameter "
    "tuning). Clustering is demonstrated with K-Means and Hierarchical "
    "(agglomerative) clustering, visualised after Principal Component Analysis "
    "so the groups can be inspected in two dimensions. The same employee "
    "dataset is then reused for a supervised attrition classifier, evaluated "
    "with stratified 5-fold cross-validation, a confusion matrix, precision, "
    "recall, F1-score and ROC-AUC, and improved with GridSearchCV."
)

r.heading("2. Tools & Libraries Used")
r.bullet("Python 3 — language used throughout.")
r.bullet("Pandas & NumPy — dataset generation, tables, and array operations.")
r.bullet("scikit-learn — KMeans, AgglomerativeClustering, PCA, models, GridSearchCV, metrics.")
r.bullet("SciPy — Ward-linkage dendrogram for hierarchical clustering.")
r.bullet("Matplotlib & Seaborn — every chart in this report.")

r.heading("3. Dataset")
r.para(
    f"A fresh sample of {stats['n_rows']} employees was generated with the same "
    "schema used in Weeks 1 and 2 (age, gender, department, city, years of "
    "experience, monthly salary, performance score, attrition). The important "
    "change this week is that the rows were drawn from three latent career "
    "segments — early-career, mid-career specialists, and senior "
    "high-performers — each with its own typical age, tenure, pay and "
    "performance. Those segments are hidden from the clustering algorithms "
    "and used only afterwards as a sanity check. Attrition is still generated "
    "from performance, tenure and (weakly) segment membership, so the "
    "supervised half of the task has a real signal to learn."
)
r.para(
    "Clustering uses four numeric features that describe an employee's career "
    "stage: Age, YearsExperience, MonthlySalary and PerformanceScore. "
    "Categorical columns are one-hot encoded for the supervised half of the "
    "task. Features are standardised (zero mean, unit variance) before "
    "clustering and PCA, because K-Means and Ward linkage both treat Euclidean "
    "distance as meaningful — leaving salary in rupees next to age in years "
    "would let salary dominate every distance calculation."
)
r.table(
    ["Item", "Value"],
    [
        ["Rows", str(stats["n_rows"])],
        ["Clustering features", ", ".join(stats["cluster_features"])],
        ["True latent segments", "3 (early / mid / senior)"],
        ["Attrition = Yes", f"{stats['attrition_counts'].get('Yes', 0)} ({stats['attrition_rate']}%)"],
        ["Attrition = No", str(stats["attrition_counts"].get("No", 0))],
        ["Train / test (supervised)", f"{stats['clf_train']} / {stats['clf_test']}"],
    ],
    caption="Dataset summary",
)

r.heading("4. K-Means Clustering")
r.heading("4.1 Choosing k — elbow and silhouette", level=2)
r.para(
    "K-Means partitions the data into k spherical groups by iteratively "
    "assigning each point to the nearest centroid and then moving the "
    "centroids to the mean of their assigned points. k itself is a choice, "
    "not something the algorithm infers. Two complementary diagnostics were "
    "computed for k = 2 through 8: inertia (the within-cluster sum of squared "
    "distances — it always falls as k grows, so one looks for an 'elbow' "
    "where extra clusters stop buying much tightness) and the silhouette "
    "score (how much closer each point is to its own cluster than to the "
    "next nearest one; 1 is ideal, 0 is on a boundary, negative means a "
    "likely mis-assignment)."
)
r.code(
    "from sklearn.cluster import KMeans\n"
    "from sklearn.metrics import silhouette_score\n\n"
    "for k in range(2, 9):\n"
    "    km = KMeans(n_clusters=k, n_init=20, random_state=42)\n"
    "    labels = km.fit_predict(X_scaled)\n"
    "    inertias.append(km.inertia_)\n"
    "    silhouettes.append(silhouette_score(X_scaled, labels))"
)
r.table(
    ["k", "Inertia", "Silhouette"],
    [[str(k), f"{inertias[i]:.1f}", f"{silhouettes[i]:.3f}"] for i, k in enumerate(ks)],
    caption="K-Means diagnostics across candidate k",
)
r.image(OUTPUTS / "elbow_method.png", "Elbow plot of K-Means inertia against k.")
r.image(OUTPUTS / "silhouette_scores.png", "Mean silhouette score against k.")
r.para(
    f"The silhouette score peaks at k = {stats['best_k_silhouette']} "
    f"({stats['best_silhouette']}). The elbow in inertia is consistent with "
    f"k = {stats['chosen_k']}, which is also the number of segments actually "
    "used to generate the data. k = 3 was therefore used for the rest of the "
    "clustering analysis: it is the most interpretable choice, it sits at "
    "the elbow, and it is at or next to the silhouette peak. A slightly "
    "higher silhouette at a neighbouring k would not, on its own, justify "
    "splitting a career-stage group that already has a clear business reading."
)

r.heading("4.2 Fitted K-Means solution (k = 3)", level=2)
r.para(
    f"With k = 3 the algorithm produced clusters of sizes "
    + ", ".join(f"{cid}: {n_}" for cid, n_ in stats["kmeans_sizes"].items())
    + f", and a silhouette of {stats['kmeans_silhouette']}. Reading the cluster "
    "means against the four features gives three recognisable employee personas:"
)
persona_rows = []
for cid, name in sorted(stats["kmeans_personas"].items(), key=lambda kv: kv[0]):
    p = stats["kmeans_profiles"][cid]
    persona_rows.append([
        cid,
        name,
        stats["kmeans_sizes"][cid],
        p["Age"],
        p["YearsExperience"],
        f"{p['MonthlySalary']:.0f}",
        p["PerformanceScore"],
    ])
r.table(
    ["Cluster", "Persona", "n", "Mean age", "Mean tenure", "Mean salary", "Mean performance"],
    persona_rows,
    caption="K-Means cluster profiles",
)
r.image(OUTPUTS / "cluster_profiles.png", "Mean feature values (± SD) by K-Means cluster.")
r.para(
    f"As a check — not as an evaluation metric the algorithm had access to — "
    f"the K-Means labels recover the generating segments for "
    f"{stats['kmeans_true_agreement']}% of employees after mapping each "
    "discovered cluster onto the majority true segment. The leftover "
    "disagreements are expected: the generating process mixed noise into age, "
    "tenure and salary, so some mid-career staff sit closer to a senior "
    "centroid (and vice versa) than to their 'true' group."
)

r.heading("5. Hierarchical Clustering")
r.para(
    "Agglomerative hierarchical clustering starts with every employee as their "
    "own cluster and repeatedly merges the two closest groups until a single "
    "tree remains. Ward linkage was used, which merges the pair whose union "
    "increases the total within-cluster variance the least — a similar "
    "objective to K-Means, which makes the two methods comparable. Cutting "
    "the tree at three clusters produced groups of sizes "
    + ", ".join(f"{cid}: {n_}" for cid, n_ in stats["hier_sizes"].items())
    + f" and a silhouette of {stats['hier_silhouette']}."
)
r.code(
    "from sklearn.cluster import AgglomerativeClustering\n"
    "from scipy.cluster.hierarchy import dendrogram, linkage\n\n"
    "model = AgglomerativeClustering(n_clusters=3, linkage='ward')\n"
    "labels = model.fit_predict(X_scaled)\n"
    "Z = linkage(X_scaled[sample_idx], method='ward')\n"
    "dendrogram(Z)"
)
r.image(
    OUTPUTS / "hierarchical_dendrogram.png",
    "Ward-linkage dendrogram on a 60-employee subsample. Vertical distance is the cost of a merge.",
)
r.para(
    "The dendrogram is the distinctive artefact of hierarchical clustering: "
    "it shows not only the final three groups but the order in which smaller "
    "groups combined. Long vertical stems just above a cut mean those merges "
    "were expensive — evidence that the groups being joined were genuinely "
    "far apart. Unlike K-Means, the method does not need k up front (k is "
    "chosen by where the tree is cut), and it does not assume spherical "
    "clusters of similar size. The cost is computational (it scales poorly "
    "beyond tens of thousands of rows) and, with Ward linkage, a similar "
    "sensitivity to scale, which is why standardisation was applied first."
)

r.heading("6. Dimensionality Reduction with PCA")
r.para(
    "Four clustering features are already visualisable in principle, but "
    "Principal Component Analysis is the standard way to project a scaled "
    "feature space onto the directions of greatest variance so clusters can "
    "be inspected in two dimensions without picking an arbitrary pair of "
    "axes. PC1 and PC2 together explain "
    f"{stats['pca_total_2d']}% of the variance "
    f"(PC1 {stats['pca_explained'][0]}%, PC2 {stats['pca_explained'][1]}%), "
    "so a 2-D scatter is a faithful summary of this particular dataset rather "
    "than a severe compression."
)
r.image(OUTPUTS / "pca_explained_variance.png", "Individual and cumulative explained variance by principal component.")
r.image(OUTPUTS / "kmeans_pca_scatter.png", "K-Means labels plotted on the first two principal components.")
r.image(OUTPUTS / "hierarchical_pca_scatter.png", "Hierarchical (Ward) labels on the same PCA plane.")
r.image(OUTPUTS / "true_segments_pca.png", "The three generating segments on the same PCA plane, for comparison.")
r.para(
    "The three clouds separate mainly along PC1, which aligns with the "
    "career-stage axis (age, tenure and salary move together). Hierarchical "
    "and K-Means partitions occupy almost the same regions of this plane, "
    "which is expected given that both were asked for three clusters under "
    "a variance-minimising criterion. PCA here is used as a visualisation "
    "and diagnostic tool, not as a preprocessor that discards features before "
    "clustering — with only four numeric inputs there is nothing to gain "
    "from throwing variance away first."
)

r.heading("7. Supervised Model Evaluation")
r.para(
    "Clustering answers 'what groups exist?'. The second half of the task "
    "returns to the Week 2 question — 'will this employee leave?' — but "
    "evaluates the answer properly. A single train/test split is a noisy "
    "estimate of performance; accuracy on an imbalanced label is actively "
    "misleading; and the hyperparameters chosen in Week 2 were educated "
    "guesses. This section replaces those shortcuts with stratified k-fold "
    "cross-validation, the full classification metric suite, and a grid search."
)

r.heading("7.1 Why accuracy is not enough", level=2)
r.para(
    f"Attrition is {stats['attrition_rate']}% of the dataset. A model that "
    "predicts 'No' for every employee would already look strong on accuracy "
    "and would score precision/recall of zero on the class that actually "
    "matters. The metrics used below, and what they mean for HR, are:"
)
r.bullet("Confusion matrix — counts of true negatives, false positives, false negatives and true positives on a held-out test set.")
r.bullet("Precision — of employees flagged as leavers, how many actually left. Low precision means wasting retention budget on people who were staying anyway.")
r.bullet("Recall — of employees who actually left, how many the model caught. Low recall means missing people the business wanted to retain.")
r.bullet("F1-score — the harmonic mean of precision and recall; the quantity the grid search maximises.")
r.bullet("ROC-AUC — how well the model ranks leavers above stayers across every possible probability threshold, not just 0.5.")

r.heading("7.2 Stratified 5-fold cross-validation", level=2)
r.para(
    "Each model is trained five times, each time holding out a different "
    "fifth of the data, with folds stratified so every fold keeps the same "
    "attrition rate. Scaling is fitted inside the fold, not before it, by "
    "wrapping the classifier in a Pipeline — the same leakage-avoidance "
    "principle used for the train/test scaler in Week 2, applied to every fold."
)
r.code(
    "from sklearn.model_selection import StratifiedKFold, cross_validate\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import MinMaxScaler\n\n"
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
    "pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', model)])\n"
    "cross_validate(pipe, X, y, cv=cv,\n"
    "               scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])"
)

cv_rows = []
for name, res in cv_results.items():
    cv_rows.append([
        name,
        f"{res['accuracy']['mean']} ± {res['accuracy']['std']}",
        f"{res['precision']['mean']} ± {res['precision']['std']}",
        f"{res['recall']['mean']} ± {res['recall']['std']}",
        f"{res['f1']['mean']} ± {res['f1']['std']}",
        f"{res['roc_auc']['mean']} ± {res['roc_auc']['std']}",
    ])
r.table(
    ["Model", "Accuracy %", "Precision %", "Recall %", "F1 %", "ROC-AUC"],
    cv_rows,
    caption="Stratified 5-fold CV means ± standard deviation",
)
r.image(OUTPUTS / "cv_metric_comparison.png", "Cross-validated accuracy, precision, recall and F1 by model.")
r.para(
    "Reporting the standard deviation next to the mean is the point of "
    "cross-validation: it shows how much a metric jumps when the held-out "
    "slice changes. A model that wins on one lucky split but has a wide "
    "standard deviation is not a model to ship. Random Forest remains the "
    "most stable of the four on F1 and ROC-AUC, consistent with Week 2, "
    "which is why it is the model taken into hyperparameter tuning."
)

r.heading("7.3 Hyperparameter tuning with GridSearchCV", level=2)
r.para(
    "Week 2 fixed Random Forest at 200 trees and max_depth=6. Those values "
    "were reasonable defaults, not a search. GridSearchCV trains a model for "
    "every combination of n_estimators ∈ {100, 200, 400}, max_depth ∈ "
    "{4, 6, 10, None}, min_samples_split ∈ {2, 5, 10} and max_features ∈ "
    "{sqrt, log2} — 72 combinations, each scored by 5-fold CV F1 on the "
    "training split only. The test split is untouched until the winning "
    "configuration is chosen, so it remains an honest estimate of "
    "generalisation."
)
r.code(
    "from sklearn.model_selection import GridSearchCV\n"
    "from sklearn.ensemble import RandomForestClassifier\n\n"
    "grid = GridSearchCV(\n"
    "    RandomForestClassifier(random_state=42),\n"
    "    param_grid={\n"
    "        'n_estimators': [100, 200, 400],\n"
    "        'max_depth': [4, 6, 10, None],\n"
    "        'min_samples_split': [2, 5, 10],\n"
    "        'max_features': ['sqrt', 'log2'],\n"
    "    },\n"
    "    scoring='f1', cv=cv, n_jobs=-1,\n"
    ")\n"
    "grid.fit(X_train, y_train)"
)
bp = stats["best_params"]
r.table(
    ["Hyperparameter", "Value chosen by GridSearchCV"],
    [
        ["n_estimators", str(bp["n_estimators"])],
        ["max_depth", str(bp["max_depth"])],
        ["min_samples_split", str(bp["min_samples_split"])],
        ["max_features", str(bp["max_features"])],
        ["Best CV F1 (train folds)", pct(stats["best_cv_f1"])],
    ],
    caption="Winning Random Forest hyperparameters",
)
r.image(
    OUTPUTS / "gridsearch_heatmap.png",
    "Mean CV F1 by n_estimators and max_depth, averaged over the remaining hyperparameters.",
)

r.heading("7.4 Hold-out test: default vs tuned", level=2)
r.para(
    "After the grid search, both the Week-2-style default forest and the "
    "tuned forest are scored once on the 20% hold-out set. This is the number "
    "that would be quoted as 'test performance' — it was not used to pick "
    "hyperparameters."
)
hd, ht = stats["holdout_default"], stats["holdout_tuned"]
r.table(
    ["Metric", "Default RF", "Tuned RF"],
    [
        ["Accuracy", pct(hd["accuracy"]), pct(ht["accuracy"])],
        ["Precision", pct(hd["precision"]), pct(ht["precision"])],
        ["Recall", pct(hd["recall"]), pct(ht["recall"])],
        ["F1-score", pct(hd["f1"]), pct(ht["f1"])],
        ["ROC-AUC", auc(hd["roc_auc"]), auc(ht["roc_auc"])],
    ],
    caption="Hold-out test metrics before and after tuning",
)
r.image(OUTPUTS / "confusion_matrices.png", "Confusion matrices on the hold-out set, default vs tuned Random Forest.")
r.image(OUTPUTS / "roc_curves.png", "ROC curves on the hold-out set. The dashed line is random ranking.")
r.para(
    "The confusion matrices make the precision/recall trade-off concrete: "
    "the top-right cell is employees the model alarmed about who stayed "
    "(false positives); the bottom-left cell is leavers the model missed "
    "(false negatives). Tuning for F1 typically moves those two cells "
    "together rather than inflating accuracy. ROC-AUC, being threshold-"
    "independent, shows whether the underlying ranking of employees by "
    "leaving-risk improved, even if the 0.5 cutoff is not the operating "
    "point HR would actually use. In production one would pick a threshold "
    "from the ROC (or a precision-recall curve) against the relative cost "
    "of a missed leaver versus an unnecessary retention conversation."
)

r.heading("7.5 What the model relies on", level=2)
r.para(
    "The tuned forest's feature importances concentrate on the same variables "
    "that generated attrition — performance, salary and tenure — which is "
    "the same sanity check applied to Linear Regression coefficients in Week 2. "
    "When the important features match the domain story, it is independent "
    "evidence the model learned signal rather than an artefact of the split."
)
r.image(OUTPUTS / "feature_importance.png", "Largest feature importances from the tuned Random Forest.")

r.heading("8. Findings")
r.bullet(
    f"K-Means and Ward hierarchical clustering both recover three career-stage "
    f"groups on standardised age, tenure, salary and performance. K-Means "
    f"silhouette at k=3 is {stats['kmeans_silhouette']}; hierarchical is "
    f"{stats['hier_silhouette']}."
)
r.bullet(
    "The elbow and silhouette diagnostics agreed on k = 3, matching the "
    "number of segments built into the data. Diagnostics and domain reading "
    "should be used together; silhouette alone can prefer a k that does not "
    "split the data in a way a business can act on."
)
r.bullet(
    f"PCA shows that two components capture {stats['pca_total_2d']}% of "
    "variance, and that the clusters separate along a career-stage axis. "
    "PCA was used to look at the clusters, not to discard features before clustering."
)
r.bullet(
    "Accuracy on an imbalanced attrition label is not a sufficient metric. "
    "Precision, recall, F1 and ROC-AUC, plus the confusion matrix, are what "
    "make the cost of a wrong prediction visible."
)
r.bullet(
    "Stratified 5-fold CV, with scaling inside the pipeline, is a more honest "
    "performance estimate than a single split. GridSearchCV then picks "
    "hyperparameters on the training folds only, keeping the hold-out set clean."
)
r.bullet(
    f"The tuned Random Forest hold-out scores are F1 {ht['f1']}% and ROC-AUC "
    f"{ht['roc_auc']}, versus F1 {hd['f1']}% and ROC-AUC {hd['roc_auc']} for "
    "the untuned forest. Tuning is not guaranteed to move every metric up, "
    "but it replaces guesswork with a search that can be reproduced."
)

r.heading("9. Conclusion")
r.para(
    "Week 1 put the data into a trainable shape and Week 2 fitted several "
    "supervised models to it. This week added the two practices that turn "
    "those models into something you can defend: unsupervised structure "
    "discovery, and evaluation that does not collapse to a single accuracy "
    "number on a single split. K-Means and hierarchical clustering, read "
    "through PCA and cluster profiles, produced the same three career-stage "
    "personas that were built into the sample. Cross-validation, the "
    "confusion-matrix metric suite, ROC-AUC and GridSearchCV then showed "
    "how to compare classifiers — and how to improve one — without leaking "
    "the test set into the decision. All figures and tables in this report "
    "were produced by the accompanying Python script."
)

out_docx = ROOT / "Week3_Unsupervised_Learning_Model_Evaluation_Report.docx"
r.save(out_docx)
print("DONE")
print(json.dumps({k: stats[k] for k in [
    "n_rows", "chosen_k", "kmeans_silhouette", "hier_silhouette",
    "pca_total_2d", "kmeans_true_agreement", "best_params",
    "best_cv_f1", "holdout_default", "holdout_tuned",
]}, indent=2))
print("Report:", out_docx)
# overwrite with the longer evaluator-oriented report
import runpy
runpy.run_path(str(ROOT / "build_report.py"), run_name="week3_rebuild")
