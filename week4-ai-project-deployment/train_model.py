"""
Week 4 — train, evaluate and serialise the attrition-risk model.

Produces:
  models/attrition_pipeline.joblib   (sklearn Pipeline via joblib)
  models/attrition_pipeline.pkl      (same object via pickle)
  models/metadata.json
  data/*.csv
  outputs/*.png
  Week4_AI_Project_Deployment_Capstone_Report.docx
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from report_utils import InternshipReport  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
for p in (DATA, MODELS, OUTPUTS):
    p.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
np.random.seed(42)
stats: dict = {}

# ---------------------------------------------------------------------------
# Data — same HR schema, attrition depends on performance / tenure / pay
# ---------------------------------------------------------------------------
n = 800
departments = ["Sales", "IT", "HR", "Finance", "Marketing"]
cities = ["New York", "Chicago", "Houston", "Phoenix", "Los Angeles"]
dept_boost = {"IT": 8000, "Finance": 6000, "Sales": 2000, "Marketing": 0, "HR": -2000}

age = np.random.randint(21, 60, n)
years_exp = np.clip((age - 21) * np.random.uniform(0.25, 0.85, n), 0, 35).round(0)
department = np.random.choice(departments, n)
gender = np.random.choice(["Male", "Female"], n)
city = np.random.choice(cities, n)
performance = np.random.randint(1, 6, n)
salary = (
    32000
    + years_exp * 950
    + performance * 2200
    + np.array([dept_boost[d] for d in department])
    + np.random.normal(0, 4000, n)
)
salary = np.clip(salary.round(2), 20000, None)
logit = (
    -1.15
    - 0.70 * (performance - 3)
    - 0.08 * (years_exp - 8)
    + 0.00002 * (45000 - salary)
    + np.random.normal(0, 0.55, n)
)
prob = 1 / (1 + np.exp(-logit))
attrition = np.where(np.random.rand(n) < prob, "Yes", "No")

df = pd.DataFrame(
    {
        "age": age.astype(int),
        "gender": gender,
        "department": department,
        "city": city,
        "years_experience": years_exp.astype(int),
        "monthly_salary": salary,
        "performance_score": performance.astype(int),
        "attrition": attrition,
    }
)
df.to_csv(DATA / "employee_attrition.csv", index=False)

numeric = ["age", "years_experience", "monthly_salary", "performance_score"]
categorical = ["gender", "department", "city"]
target = "attrition"

X = df[numeric + categorical]
y = (df[target] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.to_csv(DATA / "X_train.csv", index=False)
X_test.to_csv(DATA / "X_test.csv", index=False)

stats["n_rows"] = int(len(df))
stats["n_train"] = int(len(X_train))
stats["n_test"] = int(len(X_test))
stats["attrition_yes"] = int((y == 1).sum())
stats["attrition_no"] = int((y == 0).sum())
stats["attrition_rate"] = round(float(y.mean()) * 100, 1)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(random_state=42)),
    ]
)

param_grid = {
    "clf__n_estimators": [150, 300],
    "clf__max_depth": [6, 10, None],
    "clf__min_samples_split": [2, 6],
    "clf__max_features": ["sqrt"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
grid.fit(X_train, y_train)

best = grid.best_estimator_
stats["best_params"] = {
    k.replace("clf__", ""): (v if v is not None else "None") for k, v in grid.best_params_.items()
}
stats["best_cv_f1"] = round(float(grid.best_score_) * 100, 1)

y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
stats["test"] = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
    "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
    "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
    "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    "confusion_matrix": cm.tolist(),
}
stats["classification_report"] = classification_report(
    y_test, y_pred, target_names=["Stay", "Leave"], zero_division=0
)

# Charts
fig, ax = plt.subplots(figsize=(5.2, 4.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Hold-out Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUTS / "confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.plot(fpr, tpr, linewidth=2, label=f"Random Forest (AUC={stats['test']['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Deployed Model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUTS / "roc_curve.png")
plt.close()

ohe = best.named_steps["preprocess"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical))
feat_names = numeric + cat_names
importances = pd.Series(best.named_steps["clf"].feature_importances_, index=feat_names)
top = importances.sort_values(ascending=False).head(8)
stats["top_features"] = top.round(3).to_dict()
fig, ax = plt.subplots(figsize=(6.4, 4.4))
top.sort_values().plot(kind="barh", color="#1F4E79", ax=ax)
ax.set_title("Feature Importances — Deployed Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUTS / "feature_importance.png")
plt.close()

# Architecture diagram
fig, ax = plt.subplots(figsize=(10.2, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
boxes = [
    (0.3, 1.0, 1.8, 1.2, "Raw employee\ndata (CSV)", "#D6EAF8"),
    (2.4, 1.0, 1.8, 1.2, "Preprocess +\nRandom Forest\nPipeline", "#D5F5E3"),
    (4.5, 1.0, 1.8, 1.2, "Serialize\nJoblib / Pickle", "#FCF3CF"),
    (6.6, 1.0, 1.8, 1.2, "FastAPI\n/predict", "#FADBD8"),
    (8.3, 0.55, 1.5, 2.1, "Web UI\n+\nJSON API", "#E8DAEF"),
]
for x, yb, w, h, text, color in boxes:
    rect = plt.Rectangle((x, yb), w, h, facecolor=color, edgecolor="#1F4E79", linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, yb + h / 2, text, ha="center", va="center", fontsize=8.5, color="#1F4E79", zorder=3)
for x in (2.15, 4.25, 6.35, 8.15):
    ax.annotate("", xy=(x + 0.2, 1.6), xytext=(x - 0.15, 1.6),
                arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.6))
ax.set_title("End-to-end attrition risk application", color="#1F4E79", pad=8)
plt.tight_layout()
plt.savefig(OUTPUTS / "architecture.png")
plt.close()

# Serialise
joblib_path = MODELS / "attrition_pipeline.joblib"
pkl_path = MODELS / "attrition_pipeline.pkl"
joblib.dump(best, joblib_path)
with open(pkl_path, "wb") as f:
    pickle.dump(best, f)

example = {
    "age": 29,
    "gender": "Female",
    "department": "Sales",
    "city": "New York",
    "years_experience": 3,
    "monthly_salary": 38000,
    "performance_score": 2,
}
example_pred = int(best.predict(pd.DataFrame([example]))[0])
example_proba = float(best.predict_proba(pd.DataFrame([example]))[0, 1])
stats["example_input"] = example
stats["example_pred"] = "Yes" if example_pred == 1 else "No"
stats["example_proba"] = round(example_proba, 3)

metadata = {
    "model_type": "RandomForestClassifier inside sklearn Pipeline",
    "task": "binary classification — employee attrition (Yes=1)",
    "numeric_features": numeric,
    "categorical_features": categorical,
    "classes": {"0": "No (stay)", "1": "Yes (leave)"},
    "best_params": stats["best_params"],
    "test_metrics": stats["test"],
    "n_train": stats["n_train"],
    "n_test": stats["n_test"],
    "example_input": example,
    "serialization": {
        "joblib": "models/attrition_pipeline.joblib",
        "pickle": "models/attrition_pipeline.pkl",
    },
}
with open(MODELS / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
with open(OUTPUTS / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ---------------------------------------------------------------------------
# Word report
# ---------------------------------------------------------------------------
t = stats["test"]
r = InternshipReport(
    title="AI Project Deployment & Capstone",
    subtitle="Week 4 Task Report",
    author="Vutukuri Yaswanth Ganesh Kumar",
    date="11 September 2026",
    task="Week 4 — AI Project Deployment & Capstone",
    tools="Python, Pandas, scikit-learn, Joblib, Pickle, FastAPI, Uvicorn",
)

r.heading("1. Objective")
r.para(
    "Week 4 closes the internship by turning the attrition model developed "
    "across Weeks 2 and 3 into a complete, deployable application. The "
    "deliverable is not a notebook: it is a trained pipeline, written to disk "
    "with both Joblib and Pickle, served behind a FastAPI prediction endpoint, "
    "and wrapped in a small web UI so a non-technical HR user can score an "
    "employee without writing Python. This report documents data "
    "preprocessing, model training and testing, serialisation, the API "
    "contract, and how to run the service."
)

r.heading("2. Project architecture")
r.para(
    "Everything lives under week4-ai-project-deployment/. train_model.py "
    "builds the dataset, fits a scikit-learn Pipeline (preprocessing + "
    "Random Forest), writes the artefacts, and generates the evaluation "
    "charts used here. app.py is the FastAPI service: it loads the Joblib "
    "file once at startup and exposes GET / (the UI), GET /health, "
    "GET /metadata and POST /predict. Uvicorn hosts the app on 0.0.0.0 so it "
    "can be reached from outside the process that started it."
)
r.image(OUTPUTS / "architecture.png", "Pipeline from raw CSV through a serialised model to the FastAPI UI and JSON API.", width=6.4)

r.heading("3. Data preprocessing")
r.para(
    f"The training file contains {stats['n_rows']} synthetic employees with "
    "the same schema used all internship: age, gender, department, city, "
    "years of experience, monthly salary, performance score, and attrition. "
    f"{stats['attrition_yes']} employees left "
    f"({stats['attrition_rate']}%), so the label is imbalanced and F1 / "
    "ROC-AUC remain the metrics that matter, as argued in Week 3. Attrition "
    "is generated from performance, tenure and pay plus noise, so a tree "
    "ensemble has a genuine pattern to recover."
)
r.para(
    "Preprocessing is not applied as a one-off transformation of a CSV. It "
    "is a ColumnTransformer inside the Pipeline, so the same scaling and "
    "encoding used at train time is applied, with the same statistics, to "
    "every API request. Numeric columns are standardised; categorical "
    "columns are one-hot encoded with handle_unknown='ignore' so a new city "
    "name does not crash the service. The 80/20 split is stratified. Scaling "
    "statistics are learned from the training fold of GridSearchCV only."
)
r.code(
    "numeric = ['age', 'years_experience', 'monthly_salary', 'performance_score']\n"
    "categorical = ['gender', 'department', 'city']\n\n"
    "preprocess = ColumnTransformer([\n"
    "    ('num', StandardScaler(), numeric),\n"
    "    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),\n"
    "])\n"
    "pipe = Pipeline([\n"
    "    ('preprocess', preprocess),\n"
    "    ('clf', RandomForestClassifier(random_state=42)),\n"
    "])"
)
r.table(
    ["Split", "Rows", "Role"],
    [
        ["Full dataset", str(stats["n_rows"]), "Generated sample"],
        ["Training", str(stats["n_train"]), "Fit pipeline + GridSearchCV"],
        ["Hold-out test", str(stats["n_test"]), "Final reported metrics"],
    ],
    caption="Data splits",
)

r.heading("4. Model training and testing")
r.para(
    "Random Forest was the strongest family on this problem in Weeks 2 and 3, "
    "so it is the model that gets deployed. Hyperparameters are not copied "
    "from Week 3 blindly — they are re-selected with GridSearchCV on this "
    "week's training split, scoring F1 under stratified 5-fold CV. The "
    "winning configuration is then refit on the full training set."
)
bp = stats["best_params"]
r.table(
    ["Hyperparameter", "Value"],
    [[k, str(v)] for k, v in bp.items()] + [["Best CV F1", f"{stats['best_cv_f1']}%"]],
    caption="GridSearchCV result for the deployed forest",
)
r.table(
    ["Metric", "Hold-out test"],
    [
        ["Accuracy", f"{t['accuracy']}%"],
        ["Precision (leave)", f"{t['precision']}%"],
        ["Recall (leave)", f"{t['recall']}%"],
        ["F1-score", f"{t['f1']}%"],
        ["ROC-AUC", str(t["roc_auc"])],
    ],
    caption="Test-set performance of the serialised pipeline",
)
r.image(OUTPUTS / "confusion_matrix.png", "Confusion matrix of the deployed pipeline on the hold-out set.")
r.image(OUTPUTS / "roc_curve.png", "ROC curve of the deployed pipeline.")
r.image(OUTPUTS / "feature_importance.png", "Largest feature importances inside the deployed Random Forest.")
r.para(
    "Recall on the 'leave' class is still the hardest number, as it was in "
    "Weeks 2 and 3: most employees stay, so a 0.5 threshold is conservative. "
    "The API therefore returns the probability and a risk band (Low / Medium / "
    "High) rather than only a Yes/No label, so a caller can apply a threshold "
    "that matches the cost of a missed leaver."
)

r.heading("5. Model serialisation — Joblib and Pickle")
r.para(
    "Once the Pipeline has been fitted it is an ordinary Python object: a "
    "nested structure of transformers, a forest of decision trees, and the "
    "category levels seen during training. Serialisation writes that object "
    "to disk so the API process does not have to retrain on every startup. "
    "Two formats are saved, as the task requires:"
)
r.bullet("Joblib (models/attrition_pipeline.joblib) — the recommended way to persist scikit-learn objects. It is efficient for arrays of NumPy numeric trees and is what app.py loads.")
r.bullet("Pickle (models/attrition_pipeline.pkl) — the standard library equivalent. It produces a larger file for the same object and is kept so either loading path can be demonstrated.")
r.code(
    "import joblib, pickle\n\n"
    "joblib.dump(best, 'models/attrition_pipeline.joblib')\n"
    "with open('models/attrition_pipeline.pkl', 'wb') as f:\n"
    "    pickle.dump(best, f)\n\n"
    "# API startup\n"
    "model = joblib.load('models/attrition_pipeline.joblib')"
)
r.para(
    "A companion metadata.json records feature names, class labels and test "
    "metrics so the API can describe itself at GET /metadata without anyone "
    "opening the notebook that trained the model. The important production "
    "caveat: pickle/joblib files are Python-version and sklearn-version "
    "sensitive, and they should only be loaded from a trusted path — they "
    "can execute code on load. That is acceptable here because we both "
    "wrote and consume the file in the same project."
)

r.heading("6. Prediction API (FastAPI)")
r.para(
    "FastAPI was chosen over Flask for the same job because request bodies "
    "are validated by a Pydantic model (a missing field or a performance "
    "score of 9 is rejected with a 422 before the forest ever runs) and "
    "because /docs is generated automatically. The service binds to 0.0.0.0 "
    "and uses relative URLs in the browser UI, so it works behind a preview "
    "proxy as well as on localhost."
)
r.code(
    "POST /predict\n"
    "{\n"
    '  "age": 29,\n'
    '  "gender": "Female",\n'
    '  "department": "Sales",\n'
    '  "city": "New York",\n'
    '  "years_experience": 3,\n'
    '  "monthly_salary": 38000,\n'
    '  "performance_score": 2\n'
    "}\n\n"
    "→ {\n"
    '  "attrition": "Yes" | "No",\n'
    '  "probability_leave": 0.0–1.0,\n'
    '  "risk_level": "Low" | "Medium" | "High",\n'
    '  "message": "..."\n'
    "}"
)
r.table(
    ["Endpoint", "Method", "Purpose"],
    [
        ["/", "GET", "HR web UI (HTML)"],
        ["/presentation", "GET", "Capstone slide deck (HTML)"],
        ["/health", "GET", "Liveness check"],
        ["/metadata", "GET", "Feature list, metrics, serialisation paths"],
        ["/predict", "POST", "Score one employee"],
        ["/docs", "GET", "OpenAPI / Swagger UI"],
    ],
    caption="API surface",
)
r.para(
    f"A smoke-test of the serialised pipeline on the example employee above "
    f"returns attrition = {stats['example_pred']} with P(leave) = "
    f"{stats['example_proba']}. The same payload is what the UI sends."
)

r.heading("7. How to run")
r.code(
    "cd week4-ai-project-deployment\n"
    "python3 train_model.py          # rebuild artefacts (optional)\n"
    "uvicorn app:app --host 0.0.0.0 --port 8000"
)
r.para(
    "Open the printed origin in a browser. The home page is the predictor; "
    "/presentation is the slide deck; /docs is the interactive API. No extra "
    "frontend build step is required — FastAPI serves the HTML directly."
)

r.heading("8. Findings and limitations")
r.bullet(
    f"The deployed pipeline scores {t['roc_auc']} ROC-AUC and {t['f1']}% F1 "
    "on a hold-out set it did not see during GridSearchCV."
)
r.bullet(
    "Putting StandardScaler and OneHotEncoder inside the Pipeline is what "
    "makes deployment safe: the API cannot accidentally skip a step or use "
    "different dummy columns than training."
)
r.bullet(
    "Joblib is the file the server loads; Pickle is saved alongside it to "
    "satisfy the serialisation requirement and to show the two APIs are interchangeable for this object."
)
r.bullet(
    "This is a sample dataset with a known generating process. A production "
    "HR system would need real historical leavers, monitoring for drift, "
    "and an explicit discussion of fairness (gender and city are inputs)."
)
r.bullet(
    "The UI is intentionally small — one form, one prediction — so the "
    "deployment lesson is not buried under frontend work."
)

r.heading("9. Conclusion")
r.para(
    "The capstone is an end-to-end attrition-risk application: a documented "
    "dataset, a preprocessing-and-model Pipeline selected by cross-validated "
    "F1, artefacts written with Joblib and Pickle, and a FastAPI service that "
    "turns those artefacts into both a JSON prediction API and a usable web "
    "page. Together with Weeks 1–3 (cleaning, supervised models, clustering "
    "and evaluation) this is a complete path from a messy table of employees "
    "to a running machine-learning product. All metrics, figures and the "
    "serialised files in this submission were produced by train_model.py; "
    "the running application is app.py."
)

out_docx = ROOT / "Week4_AI_Project_Deployment_Capstone_Report.docx"
r.save(out_docx)
print("DONE")
print("best_params", stats["best_params"])
print("test", stats["test"])
print("example", stats["example_pred"], stats["example_proba"])
print("Report:", out_docx)
import runpy
runpy.run_path(str(ROOT / "build_report.py"), run_name="week4_rebuild")
