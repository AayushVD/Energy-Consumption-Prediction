"""
app.py
=============================================================================
Smart Home Energy Consumption Prediction - Streamlit Web Application
Topic: Energy Consumption Prediction using Logistic Regression
Models: model.pkl (Logistic Regression Pipeline with StandardScaler & OHE)
Dataset: dataset.csv (UCI Appliances Energy Dataset)
=============================================================================
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Home Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Paths and Model Loading
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
DATASET_FILE = os.path.join(BASE_DIR, "dataset.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(MODEL_FILE)
    # Fallback to models folder if present
    fallback_path = os.path.join(BASE_DIR, "models", "logistic_regression_pipeline.joblib")
    if os.path.exists(fallback_path):
        return joblib.load(fallback_path)
    return None

model = load_prediction_model()

# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 22px; border-radius: 12px; color: white; margin-bottom: 22px;">
    <h1 style="margin: 0; color: white; font-size: 28px;">⚡ Smart Home Energy Consumption Predictor</h1>
    <p style="margin: 8px 0 0 0; font-size: 15px; opacity: 0.9;">
        Forecasting whether household appliance energy demand will be <b>High (&gt; 60 Wh)</b> or <b>Normal / Low (&le; 60 Wh)</b>
        using a regularized Logistic Regression pipeline trained on the UCI Appliances Energy Dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Model Details & Temporal Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.info("""
**Active Model:** Logistic Regression Baseline  
**Scaler:** StandardScaler (In-Pipeline)  
**Encoding:** OneHotEncoder (`day_of_week`)  
**5-Fold CV Accuracy:** **73.99%**  
**ROC-AUC Score:** **0.8073**
""")

st.sidebar.markdown("---")
st.sidebar.header("🗓️ Temporal Variables")
hour_val = st.sidebar.slider("Hour of Day (0–23):", min_value=0, max_value=23, value=18, step=1)
day_val = st.sidebar.selectbox(
    "Day of the Week:",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    index=0
)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_analytics, tab_data, tab_about = st.tabs([
    "🔮 Live Prediction",
    "📊 Model Analytics & Plots",
    "📁 Dataset Explorer",
    "ℹ️ About & CO5 Outcome"
])

with tab_predict:
    st.subheader("🏡 Smart Home Sensor & Environmental Parameters")
    st.write("Adjust the sliders below to simulate smart home conditions and predict appliance load:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 💡 Lighting & Kitchen")
        lights_val = st.slider("Lighting Fixtures Load (Wh)", min_value=0.0, max_value=70.0, value=10.0, step=5.0)
        t1_val = st.slider("Kitchen Temperature T1 (°C)", min_value=16.0, max_value=28.0, value=21.5, step=0.1)
        rh1_val = st.slider("Kitchen Humidity RH_1 (%)", min_value=25.0, max_value=65.0, value=40.0, step=0.5)

    with col2:
        st.markdown("##### 🛋️ Living Room Conditions")
        t2_val = st.slider("Living Room Temperature T2 (°C)", min_value=16.0, max_value=30.0, value=20.5, step=0.1)
        rh2_val = st.slider("Living Room Humidity RH_2 (%)", min_value=20.0, max_value=60.0, value=40.0, step=0.5)
        wind_val = st.slider("Outside Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=4.0, step=0.2)

    with col3:
        st.markdown("##### 🌦️ Ambient Weather")
        tout_val = st.slider("Outside Temperature T_out (°C)", min_value=-6.0, max_value=30.0, value=10.0, step=0.2)
        rhout_val = st.slider("Outside Humidity RH_out (%)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)
        press_val = st.slider("Barometric Pressure (mm Hg)", min_value=725.0, max_value=775.0, value=755.0, step=0.5)

    st.markdown("---")

    input_df = pd.DataFrame([{
        'lights': float(lights_val),
        'T1': float(t1_val),
        'RH_1': float(rh1_val),
        'T2': float(t2_val),
        'RH_2': float(rh2_val),
        'T_out': float(tout_val),
        'Press_mm_hg': float(press_val),
        'RH_out': float(rhout_val),
        'Windspeed': float(wind_val),
        'hour': int(hour_val),
        'day_of_week': str(day_val)
    }])

    predict_btn = st.button("🚀 Predict Energy Consumption Status", type="primary", use_container_width=True)

    if predict_btn or True:  # Run by default on first load
        if model is None:
            st.error("Error: model.pkl could not be loaded. Please ensure model.pkl exists in the root directory.")
        else:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            prob_normal = probability[0] * 100
            prob_high = probability[1] * 100

            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background-color: #fbeae8; border-left: 6px solid #d9534f; padding: 20px; border-radius: 8px;">
                        <h2 style="color: #c9302c; margin: 0;">⚠️ HIGH ENERGY CONSUMPTION</h2>
                        <h4 style="margin: 8px 0 0 0; color: #333;">Predicted Status: Class 1 (&gt; 60 Wh)</h4>
                        <p style="margin: 8px 0 0 0; color: #555;">
                            Appliances are expected to draw heavy power during this 10-minute window.
                            Probability of high energy draw: <b>{prob_high:.1f}%</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #eaf6ea; border-left: 6px solid #5cb85c; padding: 20px; border-radius: 8px;">
                        <h2 style="color: #4cae4c; margin: 0;">✅ NORMAL / LOW CONSUMPTION</h2>
                        <h4 style="margin: 8px 0 0 0; color: #333;">Predicted Status: Class 0 (&le; 60 Wh)</h4>
                        <p style="margin: 8px 0 0 0; color: #555;">
                            Appliances are operating within baseline energy consumption.
                            Probability of normal load: <b>{prob_normal:.1f}%</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.write("##### 💡 Energy Management Recommendations:")
                if prediction == 1:
                    st.warning("""
                    - ⏰ **Peak Load Shifting:** Reschedule washing machines, dryers, and dishwashers to off-peak nocturnal hours.
                    - 💡 **Lighting Optimization:** Dim or shut down decorative and unused lighting fixtures.
                    - 🌡️ **Thermostat Moderation:** Set indoor HVAC closer to outside temperature to avoid heavy compressor cycles.
                    """)
                else:
                    st.success("""
                    - 🔋 **Optimal Usage Window:** Favorable time to run heavy electrical appliances and EV charging.
                    - 🌿 **Grid Efficient:** Household load is within baseline Energy Star consumption limits.
                    """)

            with res_col2:
                st.markdown("##### 📈 Prediction Confidence Breakdown")
                st.metric(label="High Consumption Probability", value=f"{prob_high:.1f}%")
                st.progress(float(prob_high / 100))
                st.metric(label="Normal/Low Probability", value=f"{prob_normal:.1f}%")
                st.progress(float(prob_normal / 100))

                st.caption("Powered by Scikit-Learn Logistic Regression Pipeline (`model.pkl`).")

with tab_analytics:
    st.subheader("📊 Model Performance & Diagnostic Visualizations")
    st.write("Diagnostic evaluation figures generated on the holdout test set (3,947 observations):")

    def show_plot(filename, caption):
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.image(Image.open(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Plot {filename} available in plots/ directory.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Figure 1: Class Distribution")
        show_plot("class_distribution.png", "Figure 1: Target balance thresholded at 60 Wh median.")
        st.markdown("#### Figure 3: Confusion Matrix")
        show_plot("confusion_matrix.png", "Figure 3: Confusion matrices on 3,947 holdout test samples.")

    with col_p2:
        st.markdown("#### Figure 2: Correlation Heatmap")
        show_plot("correlation_heatmap.png", "Figure 2: Pearson correlations between sensors, weather, and energy status.")
        st.markdown("#### Figure 4: ROC Curves & AUC")
        show_plot("roc_curve.png", "Figure 4: ROC curves comparing Logistic Regression, SVM, and Random Forest.")

    st.markdown("#### Figure 5: Multi-Metric Model Comparison")
    show_plot("model_comparison.png", "Figure 5: Benchmark across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.")

with tab_data:
    st.subheader("📁 Dataset Explorer (UCI Appliances Energy)")
    if os.path.exists(DATASET_FILE):
        df_sample = pd.read_csv(DATASET_FILE, nrows=500)
        st.write(f"Displaying first 500 rows of **19,735 total instances** from `dataset.csv`:")
        st.dataframe(df_sample.head(50), use_container_width=True)
        st.markdown("""
        - **Source:** [UCI Machine Learning Repository - Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/353/appliances+energy+prediction)
        - **Total Records:** 19,735 rows
        - **Logging Interval:** Every 10 minutes over 4.5 months
        - **House Location:** Low-energy residence in Stambruges, Belgium
        """)
    else:
        st.warning("dataset.csv not found in root directory.")

with tab_about:
    st.subheader("📋 Academic Project Information & CO5 Outcome")
    st.markdown("""
    ### 🎯 Course Outcome Alignment
    **CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures.**
    
    - **Real-Life Problem:** Domestic energy demand spikes stress municipal distribution transformers and inflate peak-tariff consumer electricity bills.
    - **Machine Learning Solution:** An end-to-end classification pipeline comparing **Logistic Regression** (linear baseline), **Support Vector Machine**, and **Random Forest**.
    - **Performance Evaluation:** 5-Fold Stratified Cross-Validation on the 15,788 training set, evaluated across Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices.
    
    ### 📊 Performance Summary Table:
    | Model | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Logistic Regression** | **73.99%** | **73.88%** | **71.20%** | **71.64%** | **71.42%** | **0.8073** |
    | **Support Vector Machine** | 73.72% | 73.80% | 70.92% | 72.02% | 71.47% | 0.8067 |
    | **Random Forest** | **88.41%** | **89.41%** | **87.83%** | **89.10%** | **88.46%** | **0.9562** |
    """)
