"""
Week 1 Task - Python for ML & Data Preprocessing
Generates a realistic 'raw' sample dataset (an HR/employee dataset with the
kinds of problems real data has: missing values, duplicates, inconsistent
text, outliers), then runs it through a full preprocessing pipeline with
Pandas / NumPy / scikit-learn, saving charts + a stats.json that the report
builder script will turn into the Word document.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

ASSETS = "/home/claude/assets"
os.makedirs(ASSETS, exist_ok=True)
stats = {}

# ---------------------------------------------------------------------------
# STEP 0: Build a synthetic sample dataset (stand-in for a real-world file)
# ---------------------------------------------------------------------------
np.random.seed(42)
n = 260

departments = ["Sales", "IT", "HR", "Finance", "Marketing"]
cities = ["New York", "Chicago", "Houston", "Phoenix", "Los Angeles"]
genders = ["Male", "Female"]

df = pd.DataFrame({
    "EmployeeID": range(1001, 1001 + n),
    "Age": np.random.randint(21, 60, n).astype(float),
    "Gender": np.random.choice(genders, n),
    "Department": np.random.choice(departments, n),
    "City": np.random.choice(cities, n),
    "YearsExperience": np.random.randint(0, 35, n).astype(float),
    "MonthlySalary": np.random.normal(55000, 15000, n).round(2),
    "PerformanceScore": np.random.randint(1, 6, n),
    "Attrition": np.random.choice(["Yes", "No"], n, p=[0.22, 0.78]),
})
df["MonthlySalary"] = df["MonthlySalary"].clip(lower=20000)

# missing values
for col in ["Age", "MonthlySalary", "YearsExperience", "Department"]:
    idx = np.random.choice(df.index, size=int(n * 0.06), replace=False)
    df.loc[idx, col] = np.nan

# inconsistent text casing (needs cleaning)
inconsistent_idx = np.random.choice(df.index, size=18, replace=False)
df.loc[inconsistent_idx, "City"] = df.loc[inconsistent_idx, "City"].str.lower()

# duplicate rows
df = pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)

# salary outliers
outlier_idx = df.sample(3, random_state=2).index
df.loc[outlier_idx, "MonthlySalary"] = df.loc[outlier_idx, "MonthlySalary"] * 4.5

df.to_csv("/home/claude/raw_employee_data.csv", index=False)

stats["raw_rows"] = int(df.shape[0])
stats["raw_cols"] = int(df.shape[1])
stats["cols"] = list(df.columns)

# ---------------------------------------------------------------------------
# STEP 1: Initial exploration
# ---------------------------------------------------------------------------
print("STEP 1: shape", df.shape)
missing_before_full = df.isnull().sum()
dup_count = int(df.duplicated().sum())
stats["dup_count"] = dup_count
print("duplicates:", dup_count)
print(missing_before_full)

# illustrative sample rows: show missing values + a lowercase-city row + a normal row
illus_idx = []
illus_idx += list(df[df["Age"].isnull()].index[:2])
illus_idx += list(df[df["City"].str.islower()].index[:2])
illus_idx += list(df.index[:2])
illus_idx = list(dict.fromkeys(illus_idx))[:6]
sample_cols = ["EmployeeID", "Age", "Department", "City", "MonthlySalary", "Attrition"]
sample_raw = df.loc[illus_idx, sample_cols]


def fmt(v):
    if pd.isna(v):
        return "Missing"
    if isinstance(v, float):
        return f"{v:,.2f}" if v > 1000 else f"{v:g}"
    return str(v)


stats["sample_raw_columns"] = sample_cols
stats["sample_raw_rows"] = [[fmt(v) for v in row] for row in sample_raw.values]

# charts
plt.figure(figsize=(7, 4))
missing_before_full[missing_before_full > 0].sort_values(ascending=False).plot(
    kind="bar", color="#c0392b")
plt.title("Missing Values by Column (Before Cleaning)")
plt.ylabel("Count of Missing Values")
plt.tight_layout()
plt.savefig(f"{ASSETS}/missing_values.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df["Age"].dropna(), bins=15, kde=True, color="#2980b9")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(f"{ASSETS}/age_dist.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["MonthlySalary"], color="#e67e22")
plt.title("Monthly Salary - Boxplot Before Outlier Treatment")
plt.tight_layout()
plt.savefig(f"{ASSETS}/salary_box_before.png")
plt.close()

plt.figure(figsize=(6, 4))
order = df["Department"].value_counts().index
sns.countplot(x="Department", data=df, order=order, hue="Department",
              palette="viridis", legend=False)
plt.title("Employee Count by Department")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{ASSETS}/dept_count.png")
plt.close()

plt.figure(figsize=(5, 4))
sns.countplot(x="Attrition", data=df, hue="Attrition", palette="pastel", legend=False)
plt.title("Attrition Distribution")
plt.tight_layout()
plt.savefig(f"{ASSETS}/attrition_count.png")
plt.close()

# ---------------------------------------------------------------------------
# STEP 2: Data cleaning (duplicates + text normalization)
# ---------------------------------------------------------------------------
df_clean = df.drop_duplicates().copy()
stats["rows_after_dedup"] = int(df_clean.shape[0])

city_unique_before = sorted(df["City"].unique().tolist())
df_clean["City"] = df_clean["City"].str.strip().str.title()
city_unique_after = sorted(df_clean["City"].unique().tolist())
stats["city_unique_before"] = city_unique_before
stats["city_unique_after"] = city_unique_after

# ---------------------------------------------------------------------------
# STEP 3: Handling missing values
# ---------------------------------------------------------------------------
missing_before_counts = df_clean.isnull().sum()

age_median = float(df_clean["Age"].median())
exp_median = float(df_clean["YearsExperience"].median())
sal_median = float(df_clean["MonthlySalary"].median())
dept_mode = df_clean["Department"].mode()[0]

df_clean["Age"] = df_clean["Age"].fillna(age_median)
df_clean["YearsExperience"] = df_clean["YearsExperience"].fillna(exp_median)
df_clean["MonthlySalary"] = df_clean["MonthlySalary"].fillna(sal_median)
df_clean["Department"] = df_clean["Department"].fillna(dept_mode)

missing_after_counts = df_clean.isnull().sum()

stats["age_median"] = round(age_median, 2)
stats["experience_median"] = round(exp_median, 2)
stats["salary_median"] = round(sal_median, 2)
stats["department_mode"] = dept_mode

missing_table = []
for col in ["Age", "MonthlySalary", "YearsExperience", "Department"]:
    missing_table.append({
        "column": col,
        "before": int(missing_before_counts[col]),
        "after": int(missing_after_counts[col]),
    })
stats["missing_table"] = missing_table
stats["missing_before_total"] = int(missing_before_counts.sum())
stats["missing_after_total"] = int(missing_after_counts.sum())

# ---------------------------------------------------------------------------
# STEP 4: Outlier detection & treatment (IQR method) on MonthlySalary
# ---------------------------------------------------------------------------
Q1 = df_clean["MonthlySalary"].quantile(0.25)
Q3 = df_clean["MonthlySalary"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
n_outliers = int(((df_clean["MonthlySalary"] < lower) | (df_clean["MonthlySalary"] > upper)).sum())

stats["Q1"] = round(float(Q1), 2)
stats["Q3"] = round(float(Q3), 2)
stats["IQR"] = round(float(IQR), 2)
stats["lower_bound"] = round(float(lower), 2)
stats["upper_bound"] = round(float(upper), 2)
stats["n_outliers"] = n_outliers

df_clean["MonthlySalary"] = df_clean["MonthlySalary"].clip(lower=lower, upper=upper)

plt.figure(figsize=(6, 4))
sns.boxplot(x=df_clean["MonthlySalary"], color="#27ae60")
plt.title("Monthly Salary - Boxplot After Outlier Treatment")
plt.tight_layout()
plt.savefig(f"{ASSETS}/salary_box_after.png")
plt.close()

# ---------------------------------------------------------------------------
# STEP 5: EDA - correlation heatmap
# ---------------------------------------------------------------------------
numeric_cols = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore"]
corr = df_clean[numeric_cols].corr().round(2)
stats["corr_matrix"] = {
    "columns": numeric_cols,
    "rows": [{"label": c, "values": [float(corr.loc[c, c2]) for c2 in numeric_cols]} for c in numeric_cols],
}

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------------------
# STEP 6: Feature selection
# ---------------------------------------------------------------------------
df_selected = df_clean.drop(columns=["EmployeeID"])
stats["columns_before_encoding"] = list(df_selected.columns)

# ---------------------------------------------------------------------------
# STEP 7: Encoding categorical variables
# ---------------------------------------------------------------------------
df_encoded = df_selected.copy()

le_gender = LabelEncoder()
df_encoded["Gender"] = le_gender.fit_transform(df_encoded["Gender"])
stats["gender_mapping"] = {k: int(v) for k, v in
                            zip(le_gender.classes_, le_gender.transform(le_gender.classes_))}

le_attr = LabelEncoder()
df_encoded["Attrition"] = le_attr.fit_transform(df_encoded["Attrition"])
stats["attrition_mapping"] = {k: int(v) for k, v in
                               zip(le_attr.classes_, le_attr.transform(le_attr.classes_))}

df_encoded = pd.get_dummies(df_encoded, columns=["Department", "City"],
                             prefix=["Dept", "City"], dtype=int)
stats["columns_after_encoding"] = list(df_encoded.columns)

# ---------------------------------------------------------------------------
# STEP 8: Feature scaling / normalization
# ---------------------------------------------------------------------------
scale_cols = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore"]
before_desc = df_encoded[scale_cols].describe()

scaler = MinMaxScaler()
df_final = df_encoded.copy()
df_final[scale_cols] = scaler.fit_transform(df_final[scale_cols])
after_desc = df_final[scale_cols].describe()

scale_table = []
for col in scale_cols:
    scale_table.append({
        "feature": col,
        "min_before": round(float(before_desc.loc["min", col]), 2),
        "max_before": round(float(before_desc.loc["max", col]), 2),
        "mean_before": round(float(before_desc.loc["mean", col]), 2),
        "min_after": round(float(after_desc.loc["min", col]), 2),
        "max_after": round(float(after_desc.loc["max", col]), 2),
        "mean_after": round(float(after_desc.loc["mean", col]), 2),
    })
stats["scale_table"] = scale_table

plt.figure(figsize=(6, 4))
sns.histplot(df_final["MonthlySalary"], bins=15, kde=True, color="#8e44ad")
plt.title("Monthly Salary Distribution After Min-Max Scaling")
plt.xlabel("Scaled Monthly Salary (0-1)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/salary_scaled_dist.png")
plt.close()

df_final.to_csv("/home/claude/cleaned_preprocessed_employee_data.csv", index=False)

stats["final_shape"] = list(df_final.shape)
final_preview_cols = ["Age", "YearsExperience", "MonthlySalary", "PerformanceScore", "Gender", "Attrition"]
preview = df_final[final_preview_cols].head(5).round(3)
stats["sample_final_columns"] = final_preview_cols
stats["sample_final_rows"] = [[f"{v:g}" for v in row] for row in preview.values]

with open(f"{ASSETS}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nDONE. Final shape:", df_final.shape)
print("Stats written to", f"{ASSETS}/stats.json")
