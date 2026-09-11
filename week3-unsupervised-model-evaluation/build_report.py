"""Rebuild the Week 3 Word report with extra worked examples and analysis."""
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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    silhouette_samples,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from report_utils import InternshipReport  # noqa: E402

OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

stats = json.loads((OUTPUTS / "stats.json").read_text())
raw = pd.read_csv(DATA / "week3_raw_employee_data.csv")
encoded = pd.read_csv(DATA / "week3_model_ready.csv")
feats = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore"]
X_raw = encoded[feats]
scaler = StandardScaler().fit(X_raw)
X_scaled = scaler.transform(X_raw)

km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_scaled)
labels = km.labels_
centroids = km.cluster_centers_
sil_samples = silhouette_samples(X_scaled, labels)

# three example employees closest to each centroid
examples = []
for cid in range(3):
    idx = np.where(labels == cid)[0]
    d = np.linalg.norm(X_scaled[idx] - centroids[cid], axis=1)
    i = idx[int(np.argmin(d))]
    row = raw.iloc[i]
    scaled = X_scaled[i]
    dist = {c: round(float(np.linalg.norm(scaled - centroids[c])), 3) for c in range(3)}
    examples.append({
        "id": int(row["EmployeeID"]),
        "cluster": cid,
        "persona": stats["kmeans_personas"][str(cid)],
        "age": float(row["Age"]),
        "exp": float(row["YearsExperience"]),
        "salary": float(row["MonthlySalary"]),
        "perf": int(row["PerformanceScore"]),
        "attrition": str(row["Attrition"]),
        "scaled": [round(float(v), 2) for v in scaled],
        "dist": {str(c): d for c, d in dist.items()},
        "sil": round(float(sil_samples[i]), 3),
    })
stats["examples"] = examples

# scaling illustration
stats["scale_means"] = {c: round(float(X_raw[c].mean()), 2) for c in feats}
stats["scale_stds"] = {c: round(float(X_raw[c].std()), 2) for c in feats}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
axes[0].scatter(X_raw["Age"], X_raw["MonthlySalary"], s=12, alpha=0.55, c="#2E75B6")
axes[0].set_title("Before scaling")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Monthly salary (₹)")
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 2], s=12, alpha=0.55, c=labels, cmap="viridis")
axes[1].set_title("After StandardScaler")
axes[1].set_xlabel("Age (z-score)")
axes[1].set_ylabel("Monthly salary (z-score)")
plt.tight_layout()
plt.savefig(OUTPUTS / "scaling_before_after.png")
plt.close()

# k=2 vs k=3
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(X_scaled)
for ax, k, title in [(axes[0], 2, "k = 2 (silhouette peak)"), (axes[1], 3, "k = 3 (chosen)")]:
    lab = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X_scaled)
    ax.scatter(xy[:, 0], xy[:, 1], c=lab, s=12, cmap="viridis", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(OUTPUTS / "k2_vs_k3.png")
plt.close()

# threshold sweep on hold-out using best RF (no grid)
y = encoded["Attrition"]
X = encoded.drop(columns=["Attrition", "TrueSegment"])
scale_cols = feats
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
sc = MinMaxScaler()
X_train, X_test = X_train.copy(), X_test.copy()
X_train[scale_cols] = sc.fit_transform(X_train[scale_cols])
X_test[scale_cols] = sc.transform(X_test[scale_cols])
bp = stats["best_params"]
rf = RandomForestClassifier(
    n_estimators=int(bp["n_estimators"]),
    max_depth=None if bp["max_depth"] in (None, "None") else int(bp["max_depth"]),
    min_samples_split=int(bp["min_samples_split"]),
    max_features=bp["max_features"],
    random_state=42,
)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]
thresholds = np.round(np.linspace(0.20, 0.70, 11), 2)
sweep = []
for t in thresholds:
    pred = (proba >= t).astype(int)
    sweep.append({
        "t": float(t),
        "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y_test, pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y_test, pred, zero_division=0) * 100, 1),
    })
stats["threshold_sweep"] = sweep
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot([s["t"] for s in sweep], [s["precision"] for s in sweep], marker="o", label="Precision")
ax.plot([s["t"] for s in sweep], [s["recall"] for s in sweep], marker="o", label="Recall")
ax.plot([s["t"] for s in sweep], [s["f1"] for s in sweep], marker="o", label="F1")
ax.axvline(0.5, color="gray", linestyle="--", label="Default 0.5")
ax.set_xlabel("Decision threshold P(leave)")
ax.set_ylabel("Score (%)")
ax.set_title("Precision / recall / F1 vs threshold (tuned RF)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "threshold_sweep.png")
plt.close()

# class_weight comparison
rf_bal = RandomForestClassifier(
    n_estimators=int(bp["n_estimators"]),
    max_depth=None if bp["max_depth"] in (None, "None") else int(bp["max_depth"]),
    min_samples_split=int(bp["min_samples_split"]),
    max_features=bp["max_features"],
    class_weight="balanced",
    random_state=42,
)
rf_bal.fit(X_train, y_train)
pred_bal = rf_bal.predict(X_test)
stats["balanced"] = {
    "precision": round(precision_score(y_test, pred_bal, zero_division=0) * 100, 1),
    "recall": round(recall_score(y_test, pred_bal, zero_division=0) * 100, 1),
    "f1": round(f1_score(y_test, pred_bal, zero_division=0) * 100, 1),
}

# inertia drop k=2 to k=3 vs k=3 to k=4
stats["inertia_drop_2_3"] = round(stats["inertias"][0] - stats["inertias"][1], 1)
stats["inertia_drop_3_4"] = round(stats["inertias"][1] - stats["inertias"][2], 1)

(OUTPUTS / "stats.json").write_text(json.dumps(stats, indent=2))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
hd, ht = stats["holdout_default"], stats["holdout_tuned"]
cm = ht["confusion_matrix"]
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
ex = stats["examples"]

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
    "Week 1 cleaned an employee table and Week 2 fitted supervised models to it. "
    "This week covers the two practices those reports did not yet treat with enough "
    "depth: discovering groups when there is no target (unsupervised learning) and "
    "defending a classifier with more than a single accuracy number on a single split "
    "(cross-validation, the confusion-matrix metric suite, ROC-AUC, and hyperparameter "
    "tuning). The same HR schema is reused so the story is continuous. Clustering is "
    "done with K-Means and Ward hierarchical clustering; PCA is used to look at the "
    "groups in two dimensions. Attrition classification is then re-evaluated with "
    "stratified 5-fold CV and GridSearchCV, including a worked confusion-matrix "
    "calculation, a threshold sweep, and an explicit discussion of why the tuned "
    "forest did not automatically beat the default on the hold-out set."
)
r.para(
    "The evaluator comments on Weeks 1 and 2 asked for more depth, more examples, "
    "and more detail on preprocessing and hyperparameter tuning. Those four requests "
    "shape this report: every algorithm is followed by a numerical example on a real "
    "row from the sample, preprocessing is written out with before/after numbers, and "
    "each Random Forest hyperparameter is explained before the grid search results "
    "are shown."
)

r.heading("2. Tools & Libraries Used")
r.bullet("Python 3 — language used throughout.")
r.bullet("Pandas & NumPy — dataset generation, tables, z-score arithmetic.")
r.bullet("scikit-learn — KMeans, AgglomerativeClustering, PCA, classifiers, GridSearchCV, metrics.")
r.bullet("SciPy — Ward-linkage dendrogram.")
r.bullet("Matplotlib & Seaborn — every chart.")

r.heading("3. Dataset")
r.para(
    f"The file contains {stats['n_rows']} employees with the Week 1/2 columns: "
    "EmployeeID, Age, Gender, Department, City, YearsExperience, MonthlySalary, "
    "PerformanceScore, Attrition. Rows were drawn from three latent career segments "
    "that the clustering algorithms are not shown: early-career (typically mid-20s, "
    "short tenure, lower pay), mid-career specialists, and senior high-performers. "
    f"The generating mix was {stats['true_segment_counts']['0']} / "
    f"{stats['true_segment_counts']['1']} / {stats['true_segment_counts']['2']} "
    "employees. Attrition still depends on performance, tenure and (weakly) segment, "
    f"and is imbalanced: {stats['attrition_counts']['Yes']} leavers "
    f"({stats['attrition_rate']}%) versus {stats['attrition_counts']['No']} stayers. "
    "A model that predicts Stay for everyone would already score "
    f"{round(100 - stats['attrition_rate'], 1)}% accuracy, which is why accuracy is "
    "not the headline metric in Section 8."
)
r.table(
    ["Item", "Value"],
    [
        ["Rows", str(stats["n_rows"])],
        ["Clustering features", ", ".join(stats["cluster_features"])],
        ["Latent segments (hidden)", "3 (early / mid / senior)"],
        ["Attrition = Yes", f"{stats['attrition_counts']['Yes']} ({stats['attrition_rate']}%)"],
        ["Attrition = No", str(stats["attrition_counts"]["No"])],
        ["Supervised train / test", f"{stats['clf_train']} / {stats['clf_test']}"],
    ],
    caption="Dataset summary",
)

r.heading("4. Preprocessing (worked through)")
r.para(
    "Week 2's feedback asked for the preprocessing steps to be written down rather "
    "than assumed. Clustering and PCA use Euclidean distance, so the four numeric "
    "career-stage features must be put on a common scale before any algorithm runs. "
    "Categorical columns are encoded only for the supervised half of the task."
)
r.heading("4.1 Why unscaled salary would dominate K-Means", level=2)
r.para(
    "K-Means assigns an employee to the nearest centroid using squared Euclidean "
    "distance. Age is roughly 21–60; monthly salary is roughly ₹22,000–₹1,20,000. "
    "A 5-year age gap contributes 25 to that sum of squares; a ₹5,000 salary gap "
    "contributes 25,000,000. Without scaling, the algorithm is a salary-only "
    "clustering method that ignores age, tenure and performance. StandardScaler "
    "subtracts the training mean and divides by the training standard deviation "
    "so each feature has mean 0 and variance 1."
)
r.table(
    ["Feature", "Mean (raw)", "Std (raw)", "After scaling"],
    [[c, stats["scale_means"][c], stats["scale_stds"][c], "mean 0, std 1"] for c in feats],
    caption="StandardScaler statistics fitted on the four clustering features",
)
r.para(
    f"Worked example. Employee {ex[0]['id']} is {ex[0]['age']:.0f} years old with "
    f"₹{ex[0]['salary']:,.0f} salary. Using z = (x − mean) / std:"
)
r.code(
    f"z_age    = ({ex[0]['age']:.0f} − {stats['scale_means']['Age']}) / {stats['scale_stds']['Age']}"
    f"  →  {ex[0]['scaled'][0]}\n"
    f"z_exp    = ({ex[0]['exp']:.0f} − {stats['scale_means']['YearsExperience']}) / {stats['scale_stds']['YearsExperience']}"
    f"  →  {ex[0]['scaled'][1]}\n"
    f"z_salary = ({ex[0]['salary']:.0f} − {stats['scale_means']['MonthlySalary']}) / {stats['scale_stds']['MonthlySalary']}"
    f"  →  {ex[0]['scaled'][2]}\n"
    f"z_perf   = ({ex[0]['perf']} − {stats['scale_means']['PerformanceScore']}) / {stats['scale_stds']['PerformanceScore']}"
    f"  →  {ex[0]['scaled'][3]}"
)
r.image(
    OUTPUTS / "scaling_before_after.png",
    "Age vs salary before scaling (left) and after StandardScaler (right). "
    "The left plot is a vertical salary stripe; the right plot is the space K-Means actually uses.",
)
r.heading("4.2 Encoding and the supervised split", level=2)
r.para(
    "Gender and Attrition are label-encoded (Male/Female → 0/1, No/Yes → 0/1). "
    "Department and City are one-hot encoded with drop_first=True so linear models "
    "are not given a dummy-variable trap; tree models simply ignore the redundant "
    "column. EmployeeID is dropped: it is an identifier, not a career-stage signal. "
    "TrueSegment is kept out of every model — it exists only as a post-hoc check "
    "on clustering. For classification, MinMaxScaler is fitted on the 576-row "
    "training split only and then applied to the 144-row test split. Fitting it on "
    "all 720 rows first would leak the test-set min and max into training, the "
    "mistake Week 2 already flagged. Inside cross-validation the same rule is "
    "enforced by putting the scaler in a Pipeline so each fold refits it."
)
r.code(
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import MinMaxScaler\n\n"
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, random_state=42, stratify=y)\n"
    "scaler = MinMaxScaler()\n"
    "X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])  # fit on train\n"
    "X_test[scale_cols]  = scaler.transform(X_test[scale_cols])       # apply only"
)

r.heading("5. K-Means Clustering")
r.heading("5.1 Algorithm, in one iteration", level=2)
r.para(
    "K-Means needs k in advance. It places k centroids, assigns every employee to "
    "the nearest one, then moves each centroid to the mean of its assigned points, "
    "and repeats until assignments stop changing. n_init=20 restarts the whole "
    "procedure from different random centroids and keeps the run with the lowest "
    "inertia, so a single unlucky initialisation does not define the result. "
    "random_state=42 makes that search reproducible."
)
r.heading("5.2 Choosing k — elbow, silhouette, and a business reading", level=2)
r.para(
    "Inertia (within-cluster sum of squares) always falls as k grows: more centroids "
    "can always fit the cloud more tightly. The useful signal is where the drop "
    "flattens. From k=2 to k=3 inertia falls by "
    f"{stats['inertia_drop_2_3']} (1161.9 → 653.9); from k=3 to k=4 it falls by only "
    f"{stats['inertia_drop_3_4']} (653.9 → 503.5). That is the elbow."
)
r.para(
    "Silhouette for one employee is (b − a) / max(a, b), where a is mean distance "
    "to others in the same cluster and b is mean distance to the nearest other "
    "cluster. Values near 1 mean a clear assignment; near 0 mean a boundary point; "
    "negative means the point is closer to another cluster. The mean over all "
    f"employees peaks at k = {stats['best_k_silhouette']} "
    f"({stats['best_silhouette']}), with k = 3 a close second ({stats['kmeans_silhouette']})."
)
r.table(
    ["k", "Inertia", "Δ inertia vs previous", "Silhouette"],
    [
        ["2", "1161.9", "—", "0.483"],
        ["3", "653.9", f"−{stats['inertia_drop_2_3']}", "0.472"],
        ["4", "503.5", f"−{stats['inertia_drop_3_4']}", "0.431"],
        ["5", "395.5", "−108.0", "0.438"],
        ["6", "341.3", "−54.2", "0.404"],
        ["7", "299.7", "−41.6", "0.393"],
        ["8", "271.7", "−28.0", "0.381"],
    ],
    caption="K-Means diagnostics. The large inertia drop stops after k = 3.",
)
r.image(OUTPUTS / "elbow_method.png", "Elbow plot. The steep drop ends at k = 3.")
r.image(OUTPUTS / "silhouette_scores.png", "Mean silhouette vs k. Peak at k = 2, k = 3 almost identical.")
r.image(
    OUTPUTS / "k2_vs_k3.png",
    "Same PCA plane coloured by k = 2 (left) and k = 3 (right). k = 2 just merges mid-career with one of the tails.",
)
r.para(
    "k = 3 is chosen even though k = 2 wins silhouette by 0.011. Silhouette rewards "
    "compact, well-separated blobs; merging mid-career into 'everyone else' produces "
    "two clean clouds and a slightly higher score, but it erases the persona HR can "
    "actually act on (a 36-year-old specialist is not a 26-year-old graduate). The "
    "elbow, the generating process, and interpretability all point at three groups. "
    "A metric is a vote, not a veto — this is the kind of judgement the Week 1 "
    "feedback asked to see written down rather than left implicit."
)

r.heading("5.3 Cluster profiles and three worked employees", level=2)
r.para(
    f"k = 3 yields clusters of sizes {stats['kmeans_sizes']['0']}, "
    f"{stats['kmeans_sizes']['1']} and {stats['kmeans_sizes']['2']} with mean "
    f"silhouette {stats['kmeans_silhouette']}. Sorting by mean salary names them:"
)
persona_rows = []
for cid in sorted(stats["kmeans_personas"], key=lambda c: stats["kmeans_profiles"][c]["MonthlySalary"]):
    p = stats["kmeans_profiles"][cid]
    persona_rows.append([
        cid, stats["kmeans_personas"][cid], stats["kmeans_sizes"][cid],
        p["Age"], p["YearsExperience"], f"{p['MonthlySalary']:.0f}", p["PerformanceScore"],
    ])
r.table(
    ["Cluster", "Persona", "n", "Mean age", "Mean tenure", "Mean salary (₹)", "Mean performance"],
    persona_rows,
    caption="K-Means cluster means — the three career-stage personas",
)
r.image(OUTPUTS / "cluster_profiles.png", "Mean ± SD of each feature by K-Means cluster.")
r.para(
    "Worked assignment. For each persona, the employee nearest that centroid is "
    "shown below. Distances are Euclidean in the four-dimensional scaled space. "
    "The assigned cluster is always the one with the smallest distance — that is "
    "literally the K-Means decision rule."
)
r.table(
    ["Employee", "Raw profile", "Persona", "d(C0)", "d(C1)", "d(C2)", "Silhouette"],
    [
        [
            str(e["id"]),
            f"age {e['age']:.0f}, {e['exp']:.0f} yrs, ₹{e['salary']:,.0f}, perf {e['perf']}",
            e["persona"],
            e["dist"]["0"], e["dist"]["1"], e["dist"]["2"], e["sil"],
        ]
        for e in ex
    ],
    caption="Three employees (one nearest each centroid) and their scaled distances",
)
r.para(
    f"Employee {ex[0]['id']} is closest to cluster {ex[0]['cluster']} "
    f"({ex[0]['persona']}); the next-nearest centroid is clearly further, and the "
    f"silhouette {ex[0]['sil']} is positive, so this is not a boundary case. "
    f"After mapping each discovered cluster onto the majority hidden segment, "
    f"K-Means recovers the generating label for {stats['kmeans_true_agreement']}% "
    "of rows. The remaining disagreements are employees whose noisy salary or tenure "
    "put them closer to a neighbour's centroid — expected, not a failure of the method."
)

r.heading("6. Hierarchical Clustering")
r.para(
    "Agglomerative clustering starts with 720 single-employee groups and merges the "
    "pair whose union increases total within-cluster variance the least (Ward "
    "linkage) until one tree remains. Cutting that tree at three clusters is the "
    "hierarchical analogue of setting k = 3. Unlike K-Means it does not need k "
    "before it starts, does not assume equal-sized spherical blobs, and produces a "
    "dendrogram that shows merge order. The cost is runtime (Ward is roughly "
    "quadratic in n) and, still, a dependence on scale — which is why the same "
    "StandardScaler output is reused."
)
r.code(
    "from sklearn.cluster import AgglomerativeClustering\n"
    "from scipy.cluster.hierarchy import dendrogram, linkage\n\n"
    "labels = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_scaled)\n"
    "Z = linkage(X_scaled[sample_idx], method='ward')  # 60-row subsample for a readable tree\n"
    "dendrogram(Z)"
)
r.para(
    f"The three Ward clusters have sizes {stats['hier_sizes']['0']}, "
    f"{stats['hier_sizes']['1']}, {stats['hier_sizes']['2']} and silhouette "
    f"{stats['hier_silhouette']} — within 0.02 of K-Means. On this dataset the two "
    "methods are answering almost the same variance-minimisation question, so "
    "agreement is expected. On a dataset with elongated or nested groups, Ward "
    "and K-Means would diverge, which is why both are worth running."
)
r.image(
    OUTPUTS / "hierarchical_dendrogram.png",
    "Ward dendrogram on 60 employees. A long vertical stem means an expensive merge — those two groups were far apart.",
)

r.heading("7. Dimensionality Reduction with PCA")
r.para(
    "PCA rotates the four scaled features onto axes of decreasing variance. It is "
    "used here to plot clusters, not to drop features before clustering (four "
    "columns are already cheap to cluster). PC1 captures "
    f"{stats['pca_explained'][0]}% of variance and PC2 {stats['pca_explained'][1]}%, "
    f"together {stats['pca_total_2d']}%. That is unusually high and is itself a finding: "
    "age, tenure and salary are collinear by construction (seniors are older, have "
    "worked longer, and are paid more), so almost all of the cloud is a single "
    "career-stage axis plus a smaller performance axis. A 2-D scatter is therefore "
    "a faithful view, not a cartoon."
)
r.image(OUTPUTS / "pca_explained_variance.png", "Scree plot: individual and cumulative explained variance.")
r.image(OUTPUTS / "kmeans_pca_scatter.png", "K-Means labels on PC1–PC2.")
r.image(OUTPUTS / "hierarchical_pca_scatter.png", "Ward labels on the same plane.")
r.image(OUTPUTS / "true_segments_pca.png", "Hidden generating segments on the same plane — the sanity check.")
r.para(
    "All three colourings occupy the same regions. That is the visual version of "
    f"the {stats['kmeans_true_agreement']}% agreement number: the algorithms recovered "
    "structure that was actually in the table, not an artefact of a random initialisation."
)

r.heading("8. Supervised evaluation — attrition")
r.para(
    "Clustering answers 'what groups exist?'. The rest of the report returns to "
    "'will this employee leave?', which Week 2 already modelled, but now with the "
    "evaluation the Week 2 feedback asked for: preprocessing called out above, "
    "metrics other than accuracy, cross-validation, and a real hyperparameter search."
)

r.heading("8.1 Metrics, with a worked confusion matrix", level=2)
r.para(
    f"On the 144-row hold-out set the tuned forest produced TN={tn}, FP={fp}, "
    f"FN={fn}, TP={tp}. Those four numbers are the entire classification result "
    "at threshold 0.5; every other metric is arithmetic on them."
)
r.code(
    f"accuracy  = (TP + TN) / n     = ({tp} + {tn}) / {tn+fp+fn+tp} = {ht['accuracy']}%\n"
    f"precision = TP / (TP + FP)    = {tp} / ({tp} + {fp})     = {ht['precision']}%\n"
    f"recall    = TP / (TP + FN)    = {tp} / ({tp} + {fn})    = {ht['recall']}%\n"
    f"F1        = 2PR / (P + R)     = {ht['f1']}%\n"
    f"ROC-AUC   = ranking quality across all thresholds     = {ht['roc_auc']}"
)
r.para(
    f"What this means in HR language. Of the {tp + fn} people who actually left, "
    f"the model caught {tp} and missed {fn}. Of the {tp + fp} people it flagged, "
    f"{fp} were false alarms. Missing a leaver (FN) typically costs more than an "
    "extra retention conversation (FP), so operating at 0.5 — which favours the "
    "majority Stay class — is a business choice, not a law of the algorithm. "
    "Section 8.5 sweeps that threshold."
)
r.para(
    f"A majority-class dummy that always predicts Stay would score "
    f"{round(100 - stats['attrition_rate'], 1)}% accuracy on the whole file and "
    "0% recall. That is the baseline any reported accuracy has to beat, and why "
    "this report leads with F1 and ROC-AUC."
)

r.heading("8.2 Stratified 5-fold cross-validation", level=2)
r.para(
    "A single 80/20 split is one draw. Stratified 5-fold CV trains five times, "
    "each time holding out a different fifth, keeping the 28.5% attrition rate in "
    "every fold. The mean is a more stable estimate; the standard deviation shows "
    "how much luck was in any one split. Scaling sits inside the Pipeline so fold "
    "1 cannot see fold 2's min/max."
)
cv_rows = []
for name, res in stats["cv_results"].items():
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
    caption="Stratified 5-fold CV, mean ± standard deviation",
)
r.image(OUTPUTS / "cv_metric_comparison.png", "Cross-validated accuracy, precision, recall and F1.")
r.para(
    "Reading the table. Logistic Regression and Random Forest are tied on F1 "
    f"({stats['cv_results']['Logistic Regression']['f1']['mean']} vs "
    f"{stats['cv_results']['Random Forest']['f1']['mean']}) and both beat the "
    "Decision Tree on ROC-AUC. KNN is last on every metric that is not accuracy: "
    "distance methods suffer when several one-hot columns dilute the numeric "
    "signal, even after MinMax scaling. KNN's precision standard deviation "
    f"({stats['cv_results']['K-Nearest Neighbors']['precision']['std']} points) "
    "is also the widest — it is the least trustworthy of the four. Random Forest "
    "is taken into the grid search because it matches Logistic Regression on CV "
    "F1, was the Week 2 winner, and has hyperparameters that actually move the "
    "result (a logistic model has almost nothing to tune besides C)."
)

r.heading("8.3 Hyperparameter tuning — what was searched, and why", level=2)
r.para(
    "Week 2 locked Random Forest at n_estimators=200 and max_depth=6. Those were "
    "defaults, not a search, which is exactly what the Week 2 evaluator flagged. "
    "Each hyperparameter below changes the bias–variance trade-off in a known direction:"
)
r.bullet("n_estimators {100, 200, 400} — more trees average away variance. Diminishing returns after a few hundred, but 400 is still cheap on 576 rows.")
r.bullet("max_depth {4, 6, 10, None} — 4 is a shallow, high-bias forest; None lets trees grow until leaves are pure (high variance, risk of memorising the train fold).")
r.bullet("min_samples_split {2, 5, 10} — 2 is the sklearn default (split whenever possible); 10 forbids tiny splits and regularises.")
r.bullet("max_features {sqrt, log2} — how many features each split is allowed to see. sqrt is the classification default; log2 is slightly more random, sometimes more robust.")
r.para(
    "The grid is 3 × 4 × 3 × 2 = 72 forests, each scored by 5-fold CV F1 on the "
    "training split only (360 fits). F1 is the scoring key because accuracy would "
    "have selected a model that predicts Stay. The 144-row test set is not touched "
    "until a winner is chosen."
)
r.code(
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
    "grid.fit(X_train, y_train)  # test set still unused"
)
r.table(
    ["Hyperparameter", "Search space", "Winner", "Reading"],
    [
        ["n_estimators", "100, 200, 400", str(bp["n_estimators"]), "Largest grid value — more averaging helped F1 on the folds"],
        ["max_depth", "4, 6, 10, None", str(bp["max_depth"]), "Shallow trees; the forest preferred bias over memorising 576 rows"],
        ["min_samples_split", "2, 5, 10", str(bp["min_samples_split"]), "Default; extra regularisation here did not lift F1"],
        ["max_features", "sqrt, log2", str(bp["max_features"]), "Standard classification default"],
        ["Best CV F1", "—", f"{stats['best_cv_f1']}%", "Train-fold score; not the test number"],
    ],
    caption="GridSearchCV result and interpretation of each choice",
)
r.image(
    OUTPUTS / "gridsearch_heatmap.png",
    "Mean CV F1 by n_estimators and max_depth, averaged over min_samples_split and max_features.",
)
r.para(
    "The heatmap is the extra insight a table of winners hides: F1 is relatively "
    "flat across this grid. Depth 4 with 400 trees is only a little better than "
    "depth 6 with 200 trees (the Week 2 default). That flatness is why the hold-out "
    "comparison in the next section is allowed to come out either way — a search "
    "over a plateau will not magically add ten F1 points."
)

r.heading("8.4 Hold-out: default vs tuned, including a negative result", level=2)
r.table(
    ["Metric", "Default RF (200 trees, depth 6)", "Tuned RF (grid winner)"],
    [
        ["Accuracy", f"{hd['accuracy']}%", f"{ht['accuracy']}%"],
        ["Precision", f"{hd['precision']}%", f"{ht['precision']}%"],
        ["Recall", f"{hd['recall']}%", f"{ht['recall']}%"],
        ["F1-score", f"{hd['f1']}%", f"{ht['f1']}%"],
        ["ROC-AUC", str(hd["roc_auc"]), str(ht["roc_auc"])],
    ],
    caption="Hold-out test set — unused during the grid search",
)
r.image(OUTPUTS / "confusion_matrices.png", "Confusion matrices, default vs tuned, threshold 0.5.")
r.image(OUTPUTS / "roc_curves.png", "ROC on the hold-out set. Tuning slightly improved ranking (AUC 0.812 → 0.816).")
r.para(
    f"The tuned forest is not better on hold-out F1 ({ht['f1']}% vs {hd['f1']}%) "
    f"and is essentially tied on recall ({ht['recall']}%). ROC-AUC edges up from "
    f"{hd['roc_auc']} to {ht['roc_auc']}, so the ranking of employees by risk is "
    "very slightly cleaner, but the 0.5 cutoff does not translate that into more "
    "true positives. This is the honest result, and it is more useful than a "
    "forced 'tuning helped' narrative: with 144 test rows, moving two predictions "
    "changes F1 by about a point, which is inside the CV standard deviation "
    f"({stats['cv_results']['Random Forest']['f1']['std']} points). The value of "
    "the search is that the defaults are no longer an unexamined guess, and that "
    "the search itself is reproducible."
)

r.heading("8.5 Threshold sweep — a cheaper lever than another grid", level=2)
r.para(
    "If FN is more expensive than FP, the operating threshold should be below 0.5. "
    "The table below is the same tuned forest, same test set, only the cutoff moves."
)
r.table(
    ["Threshold", "Precision %", "Recall %", "F1 %"],
    [[s["t"], s["precision"], s["recall"], s["f1"]] for s in stats["threshold_sweep"]],
    caption="Same tuned model, different decision threshold",
)
r.image(OUTPUTS / "threshold_sweep.png", "Precision falls and recall rises as the leave-threshold is lowered.")
r.para(
    "At 0.5, recall is 43.9%. Dropping the cutoff to 0.35 catches more leavers at "
    "the cost of more false alarms. That is usually the correct HR trade-off, and "
    "it did not require a larger grid. Class weighting is the other cheap lever: "
    "refitting the winning hyperparameters with class_weight='balanced' on the "
    "same split yields "
    f"precision {stats['balanced']['precision']}%, recall {stats['balanced']['recall']}%, "
    f"F1 {stats['balanced']['f1']}%. Balancing lifts recall (the forest no longer "
    "treats a missed leaver as a cheap mistake) and drops precision. Which pair "
    "is 'better' is a cost question, not a leaderboard question."
)

r.heading("8.6 What the model actually uses", level=2)
r.para(
    "Tuned-forest importances concentrate on MonthlySalary (0.286), YearsExperience "
    "(0.237), Age (0.225) and PerformanceScore (0.162). Department and city dummies "
    "are an order of magnitude smaller. That matches the generating process "
    "(attrition was a function of performance, tenure and segment, and segment is "
    "almost a function of age/tenure/pay). When the important features match the "
    "domain story, the model is not just fitting noise in a 576-row sample."
)
r.image(OUTPUTS / "feature_importance.png", "Top importances, tuned Random Forest.")

r.heading("9. Findings")
r.bullet(
    f"K-Means and Ward clustering both recover three career-stage personas on "
    f"standardised age, tenure, salary and performance (silhouettes "
    f"{stats['kmeans_silhouette']} and {stats['hier_silhouette']})."
)
r.bullet(
    "Silhouette slightly prefers k = 2; the elbow, interpretability and the "
    "generating process prefer k = 3. Metrics inform the choice of k, they do not replace it."
)
r.bullet(
    "A worked z-score and a before/after scatter show why unscaled salary would "
    "have reduced K-Means to a one-feature algorithm."
)
r.bullet(
    f"Hold-out arithmetic: TN={tn}, FP={fp}, FN={fn}, TP={tp} → precision "
    f"{ht['precision']}%, recall {ht['recall']}%, F1 {ht['f1']}%, AUC {ht['roc_auc']}."
)
r.bullet(
    "GridSearchCV over 72 Random Forest configurations made the Week 2 defaults "
    "explicit. The surface is flat: tuning did not beat the default on hold-out F1, "
    "which is reported rather than hidden."
)
r.bullet(
    "Threshold and class_weight move recall more than another pass of the same grid. "
    "Those are the knobs a deployed HR tool should expose (Week 4 does exactly that)."
)

r.heading("10. Limitations")
r.bullet("The sample is generated, so cluster recovery is a sanity check, not evidence the method finds real HR segments in a company dump.")
r.bullet("Only four numeric features were clustered. Adding encoded department would mix a categorical geometry into a Euclidean method; k-prototypes would be the fairer extension.")
r.bullet("The 144-row test set is small enough that a two-prediction swing changes F1 by a point. CV standard deviations are the right uncertainty bars.")
r.bullet("Gender and city are in the classifier. A production system would need a fairness pass before those columns are allowed to affect a retention decision.")

r.heading("11. Conclusion")
r.para(
    "This week added unsupervised structure discovery and an evaluation discipline "
    "that Week 2 did not yet apply. K-Means and hierarchical clustering, read "
    "through PCA, cluster means, and three worked employees, produced the same "
    "three career-stage personas that were built into the file. On the supervised "
    "side, preprocessing was written out with z-score arithmetic, every metric was "
    "computed from a confusion matrix by hand, 5-fold CV replaced a single split, "
    "and GridSearchCV replaced the Week 2 defaults — including the negative result "
    "that the search did not lift hold-out F1. The larger recall gain came from "
    "moving the threshold, which is the lever the Week 4 API exposes as a risk "
    "band. All figures and tables were produced by process_week3.py and this report "
    "builder."
)

path = ROOT / "Week3_Unsupervised_Learning_Model_Evaluation_Report.docx"
r.save(path)
print("Wrote", path)
