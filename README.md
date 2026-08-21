# Machine Learning Internship Tasks

Weekly deliverables for the ML internship, built in Python with Pandas, NumPy, and scikit-learn. Each week has its own folder with the script that produces every result described in that week's report, the data it generated/consumed, and the charts it produced.

## Structure

```
ml-internship-tasks/
├── week1-data-preprocessing/
│   ├── process_data.py              # cleaning, missing values, encoding, scaling, EDA
│   ├── data/                        # raw + cleaned CSVs
│   └── outputs/                     # EDA and preprocessing charts
├── week2-supervised-ml-models/
│   ├── process_week2.py             # train/test split, 6 models, evaluation
│   ├── data/                        # raw + model-ready CSVs
│   └── outputs/                     # confusion matrices, ROC curves, comparison charts
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

## Setup

```
pip install -r requirements.txt
```

## Author
Vutukuri Yaswanth Ganesh Kumar
