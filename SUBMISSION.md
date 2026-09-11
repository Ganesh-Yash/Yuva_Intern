# Submission notes — Weeks 3 and 4

GitHub URL (compulsory): https://github.com/Ganesh-Yash/Yuva_Intern

Week 3 folder: https://github.com/Ganesh-Yash/Yuva_Intern/tree/main/week3-unsupervised-model-evaluation  
Week 4 folder: https://github.com/Ganesh-Yash/Yuva_Intern/tree/main/week4-ai-project-deployment

(If the portal is reviewed before merge, use the branch URLs under `arena/01a09131-yuva-intern`.)

Paste the descriptions below (each is well over 200 words).

## Week 3 report description

This Week 3 report is written to add the depth, worked examples, preprocessing walkthrough and hyperparameter discussion that Weeks 1–2 were asked to expand. It uses a 720-row employee sample with the same HR schema as earlier weeks, but with three hidden career-stage segments (early-career, mid-career specialists, senior high-performers).

Unsupervised half. Age, tenure, salary and performance are standardised; a worked z-score for one employee is shown so it is clear why unscaled salary would have reduced K-Means to a one-feature method. K-Means is run for k = 2..8. Inertia drops 508 points from k=2 to k=3 and only 150 from k=3 to k=4 (the elbow). Silhouette peaks at k=2 (0.483) with k=3 a close second (0.472). k=3 is chosen anyway, and the report explains why a metric is a vote rather than a veto: k=2 just merges the mid-career persona HR can act on. Cluster means, a before/after scaling scatter, a k=2 vs k=3 PCA pair, and three employees nearest each centroid (with Euclidean distances to all three centres) make the assignment rule concrete. Ward hierarchical clustering is cut at the same k (silhouette 0.453) and a 60-row dendrogram shows merge cost. PCA keeps 98.3% of variance in two components because age, tenure and pay are collinear; it is used to look at clusters, not to drop features first. K-Means recovers the hidden segment for 94.3% of rows as a sanity check, not as a training target.

Supervised half. Attrition is 28.5% Yes, so a Stay-always dummy already looks strong on accuracy. The hold-out confusion matrix (TN=94, FP=9, FN=23, TP=18) is turned into precision, recall, F1 and ROC-AUC by hand. Stratified 5-fold CV with MinMax scaling inside a Pipeline is reported as mean ± sd for Logistic Regression, Decision Tree, Random Forest and KNN. Random Forest is then grid-searched over 72 combinations (n_estimators, max_depth, min_samples_split, max_features) scoring F1 on the training folds only. Each hyperparameter’s effect on bias/variance is written out. The honest result is included: the tuned forest does not beat the Week-2-style default on hold-out F1 (52.9% vs 53.7%) because the grid is flat; ROC-AUC edges 0.812 → 0.816. A threshold sweep and a class_weight='balanced' refit show that recall moves more from the cutoff than from another grid — the lever Week 4 exposes as a risk band.

Word upload: Week3_Unsupervised_Learning_Model_Evaluation_Report.docx. Script: process_week3.py.

## Week 4 report / project description

This Week 4 capstone is an end-to-end attrition-risk application, documented at the same level of preprocessing and tuning detail the Week 2 evaluator asked for. train_model.py builds an 800-row employee table (24.4% leavers), wraps StandardScaler and OneHotEncoder with a Random Forest inside one sklearn Pipeline, and selects hyperparameters with GridSearchCV on F1 (12 configurations × 5 stratified folds). The fitted Pipeline is the deployment unit: scaler means and dummy columns travel with the trees, so the API cannot skip a step. A worked request is transformed in the report — age 29 becomes z = (29 − train mean) / train std, alongside every one-hot column the forest actually sees.

Hold-out results are computed from the confusion matrix (TN=116, FP=5, FN=29, TP=10): accuracy 78.8%, precision 66.7%, recall 25.6%, F1 37.0%, ROC-AUC 0.738. Accuracy is compared to a Stay-always dummy (~75.6%) so it is not over-claimed. Recall at 0.5 is treated as a product problem, not hidden: a threshold table and Low/Medium/High bands (0.30 / 0.55) are what the API exposes so an HR lead can trade false alarms for caught leavers without retraining. Three scored profiles (junior low-performer, UI default, senior high-performer) show the band moving in the expected direction.

Serialisation uses both formats the brief names: attrition_pipeline.joblib (what Uvicorn loads) and attrition_pipeline.pkl (pickle). app.py is FastAPI on 0.0.0.0 with Pydantic validation, POST /predict, GET /health, GET /metadata, OpenAPI at /docs, an HR form at / that calls relative /predict (no localhost), and an eight-slide deck at /presentation. The Word file is the project documentation; the HTML deck is the presentation.

Word upload: Week4_AI_Project_Deployment_Capstone_Report.docx. Run: uvicorn app:app --host 0.0.0.0 --port 8000.
