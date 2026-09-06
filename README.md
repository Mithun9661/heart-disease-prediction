# Heart Disease Prediction — Healthcare Data Science Project

**Goal:** Given a patient's clinical measurements, predict whether they have heart disease. Built end-to-end on an unseen dataset (UCI Heart Disease / Cleveland) to simulate a real interview / take-home data science task.

## Dataset
- Source: UCI Machine Learning Repository (Cleveland Heart Disease dataset)
- 303 patient records, 13 clinical features + 1 target column
- Target: `1` = heart disease present, `0` = no heart disease

| Feature | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |

## Pipeline
1. **Load & Inspect** — checked shape, dtypes, missing values, duplicates
2. **Clean** — removed 1 duplicate row (303 → 302 rows), no missing values found
3. **EDA** — target balance, correlation heatmap, age/chest-pain/heart-rate patterns vs disease status
4. **Preprocessing** — train/test split (80/20, stratified), StandardScaler for numeric features
5. **Modeling** — trained and compared 3 classifiers: Logistic Regression, Random Forest, SVM (RBF kernel)
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation, confusion matrices
7. **Interpretability** — feature importance from Random Forest

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.763 | 0.879 | 0.817 | 0.865 | 0.826 ± 0.051 |
| Random Forest | 0.803 | 0.756 | 0.939 | 0.838 | **0.882** | 0.822 ± 0.044 |
| SVM (RBF) | **0.836** | 0.795 | 0.939 | **0.861** | 0.881 | 0.822 ± 0.054 |

**Best model:** Random Forest by ROC-AUC (0.882); SVM edges ahead on accuracy/F1 — worth discussing the trade-off in an interview (ROC-AUC is more robust for imbalanced/medical screening contexts where ranking risk matters).

### Top predictive features (Random Forest importance)
1. **cp** (chest pain type) — 15.7%
2. **thal** (thalassemia) — 11.8%
3. **thalach** (max heart rate) — 11.4%
4. **oldpeak** (ST depression) — 11.0%
5. **chol** (cholesterol) — 8.9%

## Key Insights (talking points for interview)
- Chest pain type is the single strongest predictor — clinically intuitive, since atypical/asymptomatic pain patterns correlate strongly with disease presence.
- Lower max heart rate achieved during exercise and higher ST depression (oldpeak) both associate with higher disease risk — consistent with known cardiology markers.
- Recall (catching true positive cases) was prioritized in model comparison since **false negatives are costlier than false positives in healthcare screening** — a missed diagnosis is worse than a false alarm that gets ruled out by further testing.
- Fasting blood sugar (fbs) and resting ECG (restecg) had the lowest importance — could be candidates for removal in a leaner production model.

## Files in this project
- `heart.csv` — raw dataset
- `heart_disease_analysis.py` — full pipeline script (reproducible, single command run); also saves `model.pkl`, `scaler.pkl`, `feature_names.pkl` for the app
- `outputs/` — all generated charts + `metrics_report.json` (structured results)
- `app.py` — Streamlit web app: enter patient details and get a live prediction, plus tabs to explore the data and compare model performance
- `requirements.txt` — pinned dependencies
- `.gitignore` — standard Python/Streamlit ignores

## How to run locally
```bash
pip install -r requirements.txt
python3 heart_disease_analysis.py   # trains models, saves charts + model.pkl
streamlit run app.py                # launches the interactive app at localhost:8501
```

## Push to GitHub
```bash
git init
git add .
git commit -m "Heart disease prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/heart-disease-prediction.git
git push -u origin main
```

## Deploy the live app (Streamlit Community Cloud — free)
1. Push the repo to GitHub (steps above) — make sure `model.pkl`, `scaler.pkl`, `feature_names.pkl`, `heart.csv`, and `outputs/metrics_report.json` are all committed (they're small, no need to .gitignore them).
2. Go to **share.streamlit.io**, sign in with GitHub.
3. Click **"New app"**, select this repo, branch `main`, and set the main file to `app.py`.
4. Click **Deploy** — you'll get a public URL like `https://<yourapp>.streamlit.app` in a couple of minutes.
5. Put this live link on your resume/LinkedIn/portfolio — much stronger than just a code repo.

## Possible extensions (good to mention if asked "what would you do next")
- Hyperparameter tuning (GridSearchCV) on Random Forest / SVM
- SHAP values for per-patient explainability
- Handle class imbalance explicitly if scaled to a larger/imbalanced dataset
- Deploy as a simple Flask/FastAPI API or Streamlit app for demo purposes
