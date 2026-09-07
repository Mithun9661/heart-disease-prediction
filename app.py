from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# ------------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_names = joblib.load(BASE_DIR / "feature_names.pkl")
    return model, scaler, feature_names


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "heart.csv")


model, scaler, feature_names = load_artifacts()
df = load_data()

st.title("❤️ Heart Disease Risk Predictor")
st.caption("A healthcare data science project — trained on the UCI Cleveland Heart Disease dataset (302 patients).")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explore the Data", "📈 Model Performance"])

# ------------------------------------------------------------------
# TAB 1: Prediction
# ------------------------------------------------------------------
with tab1:
    st.subheader("Enter patient details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"][x],
        )
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)

    with col2:
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0],
        )[1]
        restecg = st.selectbox(
            "Resting ECG Result",
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T abnormality", "LV hypertrophy"][x],
        )
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)

    with col3:
        exang = st.selectbox(
            "Exercise-Induced Angina?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0],
        )[1]
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
        )
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x],
        )

    if st.button("Predict Risk", type="primary"):
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **High risk of heart disease** — predicted probability: {prob:.1%}")
        else:
            st.success(f"✅ **Low risk of heart disease** — predicted probability: {prob:.1%}")

        st.progress(float(prob))
        st.caption("⚠️ This is a portfolio/demo model trained on a small public dataset (302 records). Not for real medical use.")

# ------------------------------------------------------------------
# TAB 2: Data Exploration
# ------------------------------------------------------------------
with tab2:
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.countplot(x="target", data=df, hue="target", palette="Set2", legend=False, ax=ax)
        ax.set_title("Target Distribution (0=No Disease, 1=Disease)")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="age", hue="target", kde=True, bins=20, palette="Set1", ax=ax)
        ax.set_title("Age Distribution by Disease Status")
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------
# TAB 3: Model Performance
# ------------------------------------------------------------------
with tab3:
    st.subheader("Model Comparison")
    metrics_path = BASE_DIR / "outputs" / "metrics_report.json"
    try:
        with metrics_path.open(encoding="utf-8") as f:
            report = json.load(f)

        results_df = pd.DataFrame(report["model_results"]).T
        st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))
        st.info(f"**Best model (by ROC-AUC):** {report['best_model']}")

        st.subheader("Feature Importance")
        fi = pd.Series(report["feature_importance"]).sort_values(ascending=True)
        fig, ax = plt.subplots()
        ax.barh(fi.index, fi.values)
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)
        plt.close(fig)
    except FileNotFoundError:
        st.info("Model performance report is not included in this deployment. Prediction and data exploration are fully available.")

st.divider()
st.caption("Built with scikit-learn + Streamlit | Dataset: UCI Heart Disease (Cleveland)")
