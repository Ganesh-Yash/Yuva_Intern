"""Rebuild the Week 4 Word report with preprocessing, tuning and worked API examples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from report_utils import InternshipReport  # noqa: E402

OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

stats = json.loads((OUTPUTS / "stats.json").read_text())
df = pd.read_csv(DATA / "employee_attrition.csv")
model = joblib.load(MODELS / "attrition_pipeline.joblib")
X_test = pd.read_csv(DATA / "X_test.csv")
# rebuild y_test from original split
from sklearn.model_selection import train_test_split

numeric = ["age", "years_experience", "monthly_salary", "performance_score"]
categorical = ["gender", "department", "city"]
X = df[numeric + categorical]
y = (df["attrition"] == "Yes").astype(int)
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
proba = model.predict_proba(X_te)[:, 1]

# class balance
fig, ax = plt.subplots(figsize=(5.2, 3.8))
counts = df["attrition"].value_counts().reindex(["No", "Yes"])
ax.bar(["Stay (No)", "Leave (Yes)"], counts.values, color=["#2E75B6", "#C0392B"])
ax.set_ylabel("Employees")
ax.set_title("Class balance in the training file")
for i, v in enumerate(counts.values):
    ax.text(i, v + 8, str(int(v)), ha="center")
plt.tight_layout()
plt.savefig(OUTPUTS / "class_balance.png")
plt.close()

# threshold sweep
sweep = []
for t in np.round(np.linspace(0.20, 0.70, 11), 2):
    pred = (proba >= t).astype(int)
    sweep.append({
        "t": float(t),
        "precision": round(precision_score(y_te, pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y_te, pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y_te, pred, zero_division=0) * 100, 1),
    })
stats["threshold_sweep"] = sweep
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot([s["t"] for s in sweep], [s["precision"] for s in sweep], marker="o", label="Precision")
ax.plot([s["t"] for s in sweep], [s["recall"] for s in sweep], marker="o", label="Recall")
ax.plot([s["t"] for s in sweep], [s["f1"] for s in sweep], marker="o", label="F1")
ax.axvline(0.5, color="gray", linestyle="--", label="API class cutoff 0.5")
ax.axvline(0.30, color="#b9770e", linestyle=":", label="Medium-risk band")
ax.axvline(0.55, color="#c0392b", linestyle=":", label="High-risk band")
ax.set_xlabel("Decision threshold P(leave)")
ax.set_ylabel("Score (%)")
ax.set_title("Deployed model: metrics vs threshold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUTS / "threshold_sweep.png")
plt.close()

# three API examples
profiles = [
    {"age": 24, "gender": "Male", "department": "Sales", "city": "Phoenix",
     "years_experience": 1, "monthly_salary": 28000, "performance_score": 1, "tag": "Junior, low performance"},
    {"age": 29, "gender": "Female", "department": "Sales", "city": "New York",
     "years_experience": 3, "monthly_salary": 38000, "performance_score": 2, "tag": "UI default example"},
    {"age": 48, "gender": "Male", "department": "IT", "city": "Chicago",
     "years_experience": 22, "monthly_salary": 92000, "performance_score": 5, "tag": "Senior high performer"},
]
api_examples = []
for p in profiles:
    row = pd.DataFrame([{k: v for k, v in p.items() if k != "tag"}])
    pr = float(model.predict_proba(row)[0, 1])
    lab = int(model.predict(row)[0])
    band = "Low" if pr < 0.30 else ("Medium" if pr < 0.55 else "High")
    api_examples.append({**p, "proba": round(pr, 3), "attrition": "Yes" if lab == 1 else "No", "band": band})
stats["api_examples"] = api_examples

# preprocessing: one row through ColumnTransformer
prep = model.named_steps["preprocess"]
raw_row = pd.DataFrame([{k: v for k, v in profiles[1].items() if k != "tag"}])
transformed = prep.transform(raw_row)
ohe = prep.named_transformers_["cat"]
feat_names = numeric + list(ohe.get_feature_names_out(categorical))
stats["preprocessed_example"] = {
    "input": {k: v for k, v in profiles[1].items() if k != "tag"},
    "output_dim": int(transformed.shape[1]),
    "feature_names": feat_names,
    "values": [round(float(v), 3) for v in np.asarray(transformed.todense() if hasattr(transformed, "todense") else transformed).ravel()],
}

# scaler stats from the fitted transformer
num = prep.named_transformers_["num"]
stats["scaler_mean"] = {c: round(float(m), 2) for c, m in zip(numeric, num.mean_)}
stats["scaler_scale"] = {c: round(float(s), 2) for c, s in zip(numeric, num.scale_)}

joblib_size = round((MODELS / "attrition_pipeline.joblib").stat().st_size / 1e6, 2)
pkl_size = round((MODELS / "attrition_pipeline.pkl").stat().st_size / 1e6, 2)
stats["file_sizes_mb"] = {"joblib": joblib_size, "pickle": pkl_size}

(OUTPUTS / "stats.json").write_text(json.dumps(stats, indent=2))

t = stats["test"]
cm = t["confusion_matrix"]
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
bp = stats["best_params"]
pe = stats["preprocessed_example"]

r = InternshipReport(
    title="AI Project Deployment & Capstone",
    subtitle="Week 4 Task Report",
    author="Vutukuri Yaswanth Ganesh Kumar",
    date="11 September 2026",
    task="Week 4 — AI Project Deployment & Capstone",
    tools="Python, Pandas, scikit-learn, Joblib, Pickle, FastAPI, Uvicorn, Pydantic",
)

r.heading("1. Objective")
r.para(
    "Week 4 is the capstone: take the attrition problem from Weeks 2–3 and ship it "
    "as an application. The deliverable is a fitted scikit-learn Pipeline (preprocessing "
    "and model together), written to disk with both Joblib and Pickle, loaded by a "
    "FastAPI service that exposes POST /predict, and wrapped in a small HR web UI. "
    "This report is the project documentation. An HTML slide deck at /presentation is "
    "the presentation. Weeks 1–2 were marked down for thin explanations and for skipping "
    "preprocessing and hyperparameter detail; those three gaps are written out here "
    "with worked rows, scaler arithmetic, a 12-point grid, a confusion-matrix calculation, "
    "and three live API examples."
)

r.heading("2. Project architecture")
r.para(
    "week4-ai-project-deployment/ is a self-contained product. train_model.py builds "
    "the 800-row sample, fits a ColumnTransformer + Random Forest Pipeline, runs "
    "GridSearchCV, writes artefacts and charts. app.py loads models/attrition_pipeline.joblib "
    "once at startup and serves GET / (UI), GET /presentation, GET /health, GET /metadata, "
    "POST /predict and the generated OpenAPI page at /docs. Uvicorn binds 0.0.0.0 so a "
    "preview proxy can reach the process. The browser never calls localhost — every fetch "
    "is a relative /predict on the same origin."
)
r.image(OUTPUTS / "architecture.png", "CSV → Pipeline → Joblib/Pickle → FastAPI → UI and JSON.", width=6.4)
r.table(
    ["Path", "Role"],
    [
        ["train_model.py", "Data, training, GridSearchCV, serialisation, charts"],
        ["app.py", "FastAPI process: load artefact, validate, predict"],
        ["models/attrition_pipeline.joblib", "What the server loads"],
        ["models/attrition_pipeline.pkl", "Same object via pickle"],
        ["models/metadata.json", "Feature list, metrics, example payload"],
        ["templates/index.html", "HR form (relative /predict)"],
        ["templates/presentation.html", "Eight-slide capstone deck"],
    ],
    caption="Repository layout for the deployed application",
)

r.heading("3. Data")
r.para(
    f"The training file has {stats['n_rows']} synthetic employees, the same schema "
    f"as the rest of the internship. {stats['attrition_yes']} left "
    f"({stats['attrition_rate']}%) and {stats['attrition_no']} stayed. Attrition is "
    "generated from performance, tenure and pay plus noise, so a tree ensemble has "
    "a real pattern to recover — but the leave class is the minority, so a dummy that "
    f"always predicts Stay already scores {round(100 - stats['attrition_rate'], 1)}% "
    "accuracy. That dummy is the baseline the deployed model has to beat on recall "
    "and F1, not on accuracy."
)
r.image(OUTPUTS / "class_balance.png", "Stay vs leave counts. Accuracy without a model is already ~75.6%.")
r.table(
    ["Split", "Rows", "How it is used"],
    [
        ["Full sample", str(stats["n_rows"]), "Generated once, seed 42"],
        ["Training", str(stats["n_train"]), "Fit ColumnTransformer + GridSearchCV"],
        ["Hold-out test", str(stats["n_test"]), "Quoted metrics; never used to pick hyperparameters"],
    ],
    caption="Data splits. Stratify=y keeps 24.4% attrition in both sides.",
)

r.heading("4. Preprocessing — inside the Pipeline, with numbers")
r.para(
    "Week 2's missing-preprocessing comment is the reason this section is long. "
    "Preprocessing is not a notebook cell that writes a second CSV. It is a "
    "ColumnTransformer stored inside the same object as the forest, so the API "
    "cannot skip a step or invent different dummy columns from the ones the trees "
    "were split on."
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
r.heading("4.1 Numeric columns — StandardScaler", level=2)
r.para(
    "Each numeric feature is transformed as z = (x − mean_train) / std_train. "
    "The means and scales below were learned from the 640 training rows only. "
    "The test set and every live API request reuse these frozen numbers; they are "
    "not recomputed on the incoming row (that would make a single request's z-score "
    "depend on who else happened to be scored in the same batch)."
)
r.table(
    ["Feature", "Train mean", "Train std", "Role"],
    [
        ["age", stats["scaler_mean"]["age"], stats["scaler_scale"]["age"], "Career stage"],
        ["years_experience", stats["scaler_mean"]["years_experience"], stats["scaler_scale"]["years_experience"], "Tenure"],
        ["monthly_salary", stats["scaler_mean"]["monthly_salary"], stats["scaler_scale"]["monthly_salary"], "Pay"],
        ["performance_score", stats["scaler_mean"]["performance_score"], stats["scaler_scale"]["performance_score"], "Last review (1–5)"],
    ],
    caption="Fitted StandardScaler statistics stored inside the Pipeline",
)
ex_in = pe["input"]
z_age = round((ex_in["age"] - stats["scaler_mean"]["age"]) / stats["scaler_scale"]["age"], 3)
r.para(
    f"Worked request. The UI default employee is age {ex_in['age']}, "
    f"{ex_in['years_experience']} years' experience, salary ₹{ex_in['monthly_salary']:,.0f}, "
    f"performance {ex_in['performance_score']}. Age becomes "
    f"({ex_in['age']} − {stats['scaler_mean']['age']}) / {stats['scaler_scale']['age']} "
    f"= {z_age}. Trees do not need scaling to split, but keeping the scaler in the "
    "Pipeline means a future swap to logistic regression or KNN does not require a "
    "new deployment path — and it is the preprocessing discipline the evaluator asked for."
)
r.heading("4.2 Categorical columns — OneHotEncoder", level=2)
r.para(
    f"Gender, department and city become {pe['output_dim'] - 4} dummy columns. "
    "handle_unknown='ignore' means a new city name produces an all-zero city block "
    "instead of crashing the API with a 500. drop_first is not used here: the forest "
    "does not suffer from collinear dummies, and keeping every level makes the "
    "feature-importance names easier to read (department_IT rather than 'not HR, "
    "not Finance, …'). After the transformer, one request is a dense vector of "
    f"{pe['output_dim']} numbers — 4 scaled numerics plus the dummies."
)
r.table(
    ["Transformed feature", "Value for the UI default employee"],
    [[n, v] for n, v in zip(pe["feature_names"], pe["values"])],
    caption="ColumnTransformer output for one POST /predict body",
)
r.heading("4.3 What is not preprocessed, and why", level=2)
r.bullet("No imputation — the generated file has no missing values. A production pipeline would add SimpleImputer inside the same ColumnTransformer so a blank salary does not 500 the API.")
r.bullet("No outlier clip on salary — Random Forest splits are robust to extreme pay; an IQR clip like Week 1 would hide genuine high-performer signal.")
r.bullet("No resampling (SMOTE, etc.) — class imbalance is handled at decision time (risk bands) rather than by fabricating leavers.")
r.bullet("Identifiers are not in the schema. Nothing like EmployeeID can leak into a split.")

r.heading("5. Training, hyperparameter tuning, testing")
r.para(
    "Random Forest is the model that ships because it won or tied Weeks 2 and 3 on "
    "this schema and because a Pipeline of trees serialises cleanly. Hyperparameters "
    "are not copied from Week 3. They are re-searched on this week's 640-row training "
    "split, scoring F1 under stratified 5-fold CV."
)
r.heading("5.1 What each hyperparameter does", level=2)
r.bullet("n_estimators {150, 300} — number of trees. 300 costs a larger Joblib file (see §6) but averages more.")
r.bullet("max_depth {6, 10, None} — 6 is the Week 2 default (regularised); None lets trees grow until leaves are pure.")
r.bullet("min_samples_split {2, 6} — 2 is unrestricted splits; 6 is mild regularisation.")
r.bullet("max_features {sqrt} — classification default, held fixed so the grid stays 2×3×2×1 = 12 configurations × 5 folds = 60 fits.")
r.para(
    "F1 is the scoring key because accuracy would have selected a Stay-always "
    "behaviour. The 160-row test set is locked until the search finishes."
)
r.table(
    ["Hyperparameter", "Grid", "Winner", "Insight"],
    [
        ["n_estimators", "150, 300", str(bp["n_estimators"]), "More trees won — extra averaging helped minority-class F1 on the folds"],
        ["max_depth", "6, 10, None", str(bp["max_depth"]), "Unrestricted depth; 640 rows are enough that the forest wanted more leaf purity"],
        ["min_samples_split", "2, 6", str(bp["min_samples_split"]), "Default; extra regularisation did not lift CV F1"],
        ["max_features", "sqrt", str(bp["max_features"]), "Held constant"],
        ["Best CV F1", "—", f"{stats['best_cv_f1']}%", "Train-fold score, not the test number"],
    ],
    caption="GridSearchCV on the deployed Pipeline",
)
r.para(
    f"CV F1 of {stats['best_cv_f1']}% looks low next to 78.8% accuracy. That is the "
    "point of the metric: the leave class is 24.4% of the file, noisy, and hard. "
    "A 25% CV F1 is a model that has some ranking skill (confirmed by ROC-AUC below) "
    "but should not be blindly trusted at a 0.5 cutoff — which is why the API returns "
    "a probability and a band, not only Yes/No."
)

r.heading("5.2 Hold-out test, worked from the confusion matrix", level=2)
r.para(
    f"The 160-row test set produced TN={tn}, FP={fp}, FN={fn}, TP={tp}."
)
r.code(
    f"accuracy  = (TP+TN)/n      = ({tp}+{tn})/{tn+fp+fn+tp} = {t['accuracy']}%\n"
    f"precision = TP/(TP+FP)     = {tp}/({tp}+{fp})      = {t['precision']}%\n"
    f"recall    = TP/(TP+FN)     = {tp}/({tp}+{fn})     = {t['recall']}%\n"
    f"F1        = 2PR/(P+R)      = {t['f1']}%\n"
    f"ROC-AUC   = ranking skill  = {t['roc_auc']}"
)
r.table(
    ["Metric", "Hold-out", "What it means for HR"],
    [
        ["Accuracy", f"{t['accuracy']}%", f"Looks strong, but a Stay-always dummy already gets ~{round(100-stats['attrition_rate'],1)}%"],
        ["Precision (leave)", f"{t['precision']}%", f"Of {tp+fp} flagged people, {tp} really left — {fp} wasted conversations"],
        ["Recall (leave)", f"{t['recall']}%", f"Of {tp+fn} real leavers, {fn} were missed at threshold 0.5"],
        ["F1", f"{t['f1']}%", "Balances the two; this is what GridSearchCV maximised"],
        ["ROC-AUC", str(t["roc_auc"]), "0.74 means the forest ranks a random leaver above a random stayer 74% of the time"],
    ],
    caption="Test metrics of the serialised Pipeline",
)
r.image(OUTPUTS / "confusion_matrix.png", "Hold-out confusion matrix at the 0.5 class cutoff the API uses for attrition=Yes/No.")
r.image(OUTPUTS / "roc_curve.png", "ROC of the deployed Pipeline. Threshold-independent ranking is stronger than the 0.5 F1 suggests.")
r.image(OUTPUTS / "feature_importance.png", "Importances: salary, age, tenure, performance dominate; city/department are small.")
r.para(
    "The importance order matches the generating process, the same sanity check used "
    "on Linear Regression coefficients in Week 2 and on the Week 3 forest. City and "
    "gender still appear with small weights; a production fairness review would test "
    "whether dropping them changes recall materially before they are allowed to "
    "influence a retention decision."
)

r.heading("5.3 Thresholds and risk bands — the lever the API actually exposes", level=2)
r.para(
    f"Recall of {t['recall']}% at 0.5 is the same problem Weeks 2 and 3 had. Rather "
    "than pretend another grid will fix it, the application treats 0.5 as one cutoff "
    "among many. The class label attrition=Yes uses 0.5; the risk band uses 0.30 and "
    "0.55 so a caller who cares about FN can act on Medium without waiting for Yes."
)
r.table(
    ["Threshold", "Precision %", "Recall %", "F1 %"],
    [[s["t"], s["precision"], s["recall"], s["f1"]] for s in stats["threshold_sweep"]],
    caption="Same serialised model, only the cutoff moves",
)
r.image(
    OUTPUTS / "threshold_sweep.png",
    "Precision/recall/F1 vs threshold. Dotted lines are the Medium (0.30) and High (0.55) risk-band edges.",
)
r.para(
    "Lowering the cutoff from 0.5 toward 0.3 roughly doubles recall and spends "
    "precision to do it. That is the conversation an HR lead should have — 'how many "
    "false alarms will we tolerate to catch another ten leavers?' — and it is a "
    "one-line change in app.py, not a retrain."
)

r.heading("6. Serialisation — Joblib and Pickle")
r.para(
    "A fitted Pipeline is a Python object: the scaler means, the one-hot category "
    "lists, and ~300 trees. Serialisation writes that object to disk so Uvicorn does "
    "not retrain on every boot."
)
r.bullet(f"Joblib — models/attrition_pipeline.joblib ({stats['file_sizes_mb']['joblib']} MB). Efficient for NumPy arrays of trees. This is what get_model() loads.")
r.bullet(f"Pickle — models/attrition_pipeline.pkl ({stats['file_sizes_mb']['pickle']} MB). Standard-library equivalent of the same object, submitted because the brief asks for both.")
r.code(
    "import joblib, pickle\n"
    "joblib.dump(best, 'models/attrition_pipeline.joblib')\n"
    "pickle.dump(best, open('models/attrition_pipeline.pkl', 'wb'))\n\n"
    "# app.py startup (once per process)\n"
    "model = joblib.load('models/attrition_pipeline.joblib')"
)
r.para(
    "Caveats that belong in a capstone, not a footnote. (1) Both formats are tied to "
    "the sklearn/numpy version that wrote them — bumping scikit-learn without "
    "retraining can fail on load. (2) Loading a pickle executes bytecode; only files "
    "this repo wrote are loaded. (3) metadata.json sits beside them so GET /metadata "
    "can describe features and test metrics without unpickling."
)

r.heading("7. Prediction API and three worked requests")
r.para(
    "FastAPI over Flask: Pydantic rejects a performance_score of 9 with HTTP 422 "
    "before the forest runs, and /docs is free. CORS allows any origin; credentials "
    "are off, which is the combination that is actually valid with allow_origins=['*']."
)
r.table(
    ["Endpoint", "Method", "Purpose"],
    [
        ["/", "GET", "HR web UI"],
        ["/presentation", "GET", "Capstone slides"],
        ["/health", "GET", "Liveness + model file present"],
        ["/metadata", "GET", "Features, metrics, serialisation paths"],
        ["/predict", "POST", "Score one employee"],
        ["/docs", "GET", "OpenAPI / Swagger"],
    ],
    caption="API surface of app.py",
)
r.para("Three requests through the serialised Pipeline (the same code path as POST /predict):")
r.table(
    ["Profile", "Key fields", "P(leave)", "Label", "Risk band"],
    [
        [e["tag"],
         f"age {e['age']}, {e['years_experience']} yrs, ₹{e['monthly_salary']:,}, perf {e['performance_score']}",
         e["proba"], e["attrition"], e["band"]]
        for e in api_examples
    ],
    caption="Worked predictions from the Joblib artefact",
)
r.para(
    "The junior low-performer sits in a higher band than the senior high-performer, "
    "which is the direction the generating process promised. The UI default "
    f"({api_examples[1]['proba']} probability, band {api_examples[1]['band']}) is "
    "deliberately near the Medium boundary so a reviewer can see the band move by "
    "editing one field. Response body:"
)
r.code(
    "{\n"
    '  "attrition": "Yes" | "No",          // 0.5 cutoff\n'
    '  "probability_leave": 0.0-1.0,       // raw forest vote share\n'
    '  "risk_level": "Low" | "Medium" | "High",  // 0.30 / 0.55\n'
    '  "message": "recommended next step"\n'
    "}"
)

r.heading("8. How to run")
r.code(
    "cd week4-ai-project-deployment\n"
    "python3 train_model.py                 # optional rebuild\n"
    "uvicorn app:app --host 0.0.0.0 --port 8000"
)
r.para(
    "Open / for the form, /presentation for the deck, /docs to POST interactively. "
    "There is no frontend build. Bind address 0.0.0.0 is required off localhost; "
    "127.0.0.1 would make the live preview unreachable."
)

r.heading("9. Findings, limitations, next steps")
r.bullet(
    f"Deployed test scores: accuracy {t['accuracy']}%, precision {t['precision']}%, "
    f"recall {t['recall']}%, F1 {t['f1']}%, ROC-AUC {t['roc_auc']}."
)
r.bullet(
    "The Pipeline is the deployment unit. Scaling and one-hot statistics travel with "
    "the trees, which is the preprocessing story Week 2 was missing."
)
r.bullet(
    f"Joblib ({stats['file_sizes_mb']['joblib']} MB) is what the server loads; pickle "
    f"({stats['file_sizes_mb']['pickle']} MB) is the same object in the standard library format."
)
r.bullet(
    "Recall at 0.5 is weak because leavers are 24% of the file. The product response "
    "is risk bands and a threshold table, not a claim that GridSearchCV solved class imbalance."
)
r.bullet(
    "Limitations: synthetic data; gender and city still in the model; no drift monitor; "
    "pickle/joblib are version-sensitive; a real HR tool needs an appeal path and a "
    "fairness review before scores affect anyone's career."
)
r.bullet(
    "Next: SimpleImputer in the ColumnTransformer, class_weight='balanced' as a second "
    "artefact, and logging of /predict inputs (without names) to watch for drift."
)

r.heading("10. Conclusion")
r.para(
    "The capstone is a running attrition-risk product: a documented sample, a "
    "ColumnTransformer + Random Forest Pipeline whose hyperparameters were searched "
    "on F1, artefacts in Joblib and Pickle, and a FastAPI service that turns those "
    "artefacts into JSON and a form. Preprocessing is no longer an implied notebook "
    "step — the scaler means, the dummy columns, and a worked request are in this "
    "report. Tuning is no longer an implied default — the grid, the winner, the "
    "hold-out confusion matrix, and the threshold sweep are in this report. Together "
    "with Weeks 1–3 this is a full path from a messy employee table to a service a "
    "person can call. train_model.py produced the artefacts; app.py is the service; "
    "/presentation is the slide deck that accompanies this document."
)

path = ROOT / "Week4_AI_Project_Deployment_Capstone_Report.docx"
r.save(path)
print("Wrote", path)
print("api_examples", api_examples)
