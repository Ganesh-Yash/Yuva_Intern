# Week 4 — AI Project Deployment & Capstone

End-to-end attrition-risk application: sklearn Pipeline, Joblib/Pickle serialisation, FastAPI JSON API, web UI, and HTML presentation.

## Train (rebuild artefacts)

```
python3 train_model.py
```

Writes `models/attrition_pipeline.joblib`, `models/attrition_pipeline.pkl`, charts, and `Week4_AI_Project_Deployment_Capstone_Report.docx`.

## Run the API

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

| Path | What it is |
| --- | --- |
| `/` | Predictor UI |
| `/presentation` | Capstone slides |
| `/predict` | POST JSON prediction |
| `/docs` | OpenAPI |
| `/health` | Liveness |
| `/metadata` | Feature list and test metrics |
