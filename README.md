# Machine Learning Internship Tasks

Weekly deliverables for the ML internship, built in Python with Pandas, NumPy, and scikit-learn. Each week has its own folder with the script that produces every result described in that week's report, the data it generated/consumed, and the charts it produced.

## Structure

```
Yuva_Intern/
├── week1-data-preprocessing/
│   ├── process_data.py
│   ├── data/
│   └── outputs/
├── week2-supervised-ml-models/
│   ├── process_week2.py
│   ├── data/
│   └── outputs/
├── week3-unsupervised-model-evaluation/
│   ├── process_week3.py              # K-Means, hierarchical, PCA, CV, GridSearchCV
│   ├── data/
│   └── outputs/
├── week4-ai-project-deployment/
│   ├── train_model.py                # pipeline + joblib/pickle artefacts
│   ├── app.py                        # FastAPI predictor + UI
│   ├── models/
│   └── templates/
├── report_utils.py                   # shared Word-report helper
└── requirements.txt
```

## Week 1 — Python for Machine Learning & Data Preprocessing
Generates a sample employee dataset with realistic data-quality issues (missing values, duplicates, inconsistent text, outliers) and walks it through loading, cleaning, missing-value imputation, IQR outlier treatment, feature selection, encoding, and Min-Max normalization.

```
cd week1-data-preprocessing
python3 process_data.py
```

## Week 2 — Supervised Machine Learning Models
Builds on Week 1's pipeline with a dataset where Salary and Attrition genuinely depend on the other features, then trains and compares:
- **Classification** (predict Attrition): Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
- **Regression** (predict Monthly Salary): Linear Regression, Decision Tree Regressor

```
cd week2-supervised-ml-models
python3 process_week2.py
```

## Week 3 — Unsupervised Learning & Model Evaluation
Clusters the employee sample with **K-Means** (elbow + silhouette) and **hierarchical (Ward)** clustering, visualises groups with **PCA**, then evaluates attrition classifiers with **stratified 5-fold CV**, confusion matrices, precision / recall / F1 / ROC-AUC, and **GridSearchCV**.

```
cd week3-unsupervised-model-evaluation
python3 process_week3.py
```

## Week 4 — AI Project Deployment & Capstone
Trains a preprocessing + Random Forest **Pipeline**, serialises it with **Joblib and Pickle**, and serves **POST /predict** plus an HR web UI from **FastAPI**.

```
cd week4-ai-project-deployment
python3 train_model.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `/` for the predictor, `/presentation` for the slide deck, `/docs` for OpenAPI.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Author
Vutukuri Yaswanth Ganesh Kumar
