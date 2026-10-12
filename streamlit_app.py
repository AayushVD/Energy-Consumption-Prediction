"""
streamlit_app.py
=============================================================================
Interactive Web Application for Energy Consumption Prediction
Topic: Energy Consumption Prediction using Logistic Regression (and Comparison Models)
=============================================================================
"""

import os
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
# Load Models
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

@st.cache_resource
def load_models():
    lr_path = os.path.join(MODELS_DIR, "logistic_regression_pipeline.joblib")
    rf_path = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
    lr_pipe = joblib.load(lr_path) if os.path.exists(lr_path) else None
    rf_pipe = joblib.load(rf_path) if os.path.exists(rf_path) else None
    return lr_pipe, rf_pipe

lr_pipeline, rf_pipeline = load_models()

# -----------------------------------------------------------------------------
# Header and Overview
# -----------------------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 24px; border-radius: 12px; color: white; margin-bottom: 24px;">
    <h1 style="margin: 0; color: white;">⚡ Smart Home Energy Consumption Predictor</h1>
    <p style="margin: 8px 0 0 0; font-size: 16px; opacity: 0.9;">
        Predict whether household appliance energy demand will be <b>High (&gt; 60 Wh)</b> or <b>Normal / Low (&le; 60 Wh)</b>
        using machine learning pipelines built on the UCI Appliances Energy Dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Model Selection and Quick Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Algorithm:",
    ["Logistic Regression (Baseline)", "Random Forest (Ensemble)"],
    index=0,
    help="Logistic Regression offers linear probabilistic interpretability; Random Forest captures non-linear interactions."
)

st.sidebar.markdown("---")
st.sidebar.header("🗓️ Temporal Variables")
hour_val = st.sidebar.slider("Hour of Day (0–23):", min_value=0, max_value=23, value=18, step=1)
day_val = st.sidebar.selectbox(
    "Day of the Week:",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    index=0
)

# -----------------------------------------------------------------------------
# Main Tabs: Predictor, Feature Insights, Evaluation Plots
# -----------------------------------------------------------------------------
tab_predict, tab_analytics, tab_about = st.tabs(["🔮 Live Prediction", "📊 Visual Analytics & Plots", "ℹ️ About & CO5 Mapping"])

with tab_predict:
    st.subheader("🏡 Environmental & Sensor Input Parameters")
    st.write("Adjust the sliders below to simulate smart home conditions and predict energy demand:")

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
        st.markdown("##### 🌦️ Ambient Outdoor Weather")
        tout_val = st.slider("Outside Temperature T_out (°C)", min_value=-6.0, max_value=30.0, value=10.0, step=0.2)
        rhout_val = st.slider("Outside Humidity RH_out (%)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)
        press_val = st.slider("Barometric Pressure (mm Hg)", min_value=725.0, max_value=775.0, value=755.0, step=0.5)

    st.markdown("---")

    # Construct DataFrame for inference
    input_data = pd.DataFrame([{
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

    if predict_btn:
        active_pipe = lr_pipeline if "Logistic" in model_choice else rf_pipeline

        if active_pipe is None:
            st.error("Model file not found. Please run 'python train.py' first.")
        else:
            prediction = active_pipe.predict(input_data)[0]
            probability = active_pipe.predict_proba(input_data)[0]

            prob_low = probability[0] * 100
            prob_high = probability[1] * 100

            st.write("")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background-color: #fbeae8; border-left: 6px solid #d9534f; padding: 20px; border-radius: 8px;">
                        <h2 style="color: #c9302c; margin: 0;">⚠️ HIGH ENERGY CONSUMPTION</h2>
                        <h4 style="margin: 8px 0 0 0; color: #333;">Predicted Status: Class 1 (&gt; 60 Wh)</h4>
                        <p style="margin: 8px 0 0 0; color: #555;">
                            Appliances are expected to draw significant power during this period.
                            Estimated probability of peak/high load: <b>{prob_high:.1f}%</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #eaf6ea; border-left: 6px solid #5cb85c; padding: 20px; border-radius: 8px;">
                        <h2 style="color: #4cae4c; margin: 0;">✅ NORMAL / LOW CONSUMPTION</h2>
                        <h4 style="margin: 8px 0 0 0; color: #333;">Predicted Status: Class 0 (&le; 60 Wh)</h4>
                        <p style="margin: 8px 0 0 0; color: #555;">
                            Appliances are operating at baseline energy efficiency.
                            Estimated probability of normal load: <b>{prob_low:.1f}%</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.write("##### 💡 Energy Management Recommendations:")
                if prediction == 1:
                    st.warning("""
                    - ⏰ **Peak Load Shifting:** Reschedule heavy laundry or dishwashing to off-peak nocturnal hours.
                    - 💡 **Lighting Optimization:** Dim or turn off non-essential lighting fixtures.
                    - 🌡️ **Thermostat Moderation:** Align indoor HVAC setpoints closer to ambient outdoor temperature.
                    """)
                else:
                    st.success("""
                    - 🔋 **Favorable Window:** Ideal time for running laundry machines, EV charging, or high-draw appliances.
                    - 🌿 **Grid Friendly:** Household load is within energy-star efficiency boundaries.
                    """)

            with res_col2:
                st.markdown("##### 📈 Prediction Confidence Breakdown")
                st.metric(label="High Consumption Probability", value=f"{prob_high:.1f}%")
                st.progress(float(prob_high / 100))
                st.metric(label="Normal/Low Probability", value=f"{prob_low:.1f}%")
                st.progress(float(prob_low / 100))

                st.caption(f"Inference generated via **{model_choice}** with StandardScaler Pipeline.")

with tab_analytics:
    st.subheader("📊 Model Performance & Diagnostic Visualizations")
    st.write("Below are the 5 evaluation figures generated by the 5-fold cross-validated pipeline:")

    chart_select = st.selectbox(
        "Select Figure to Display:",
        [
            "All Figures",
            "1. Class Distribution",
            "2. Correlation Heatmap",
            "3. Confusion Matrix (LR vs RF)",
            "4. ROC Curves & AUC Comparison",
            "5. Overall Model Comparison Bar Chart"
        ]
    )

    def show_plot(filename, caption):
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.image(Image.open(path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Plot {filename} not found.")

    if chart_select == "All Figures" or "Class Distribution" in chart_select:
        st.markdown("#### Figure 1: Energy Consumption Class Distribution")
        show_plot("class_distribution.png", "Figure 1: Target distribution split at the median (60 Wh threshold)")

    if chart_select == "All Figures" or "Correlation Heatmap" in chart_select:
        st.markdown("#### Figure 2: Feature Correlation Heatmap")
        show_plot("correlation_heatmap.png", "Figure 2: Pearson correlation coefficients between sensors, weather, and energy status")

    if chart_select == "All Figures" or "Confusion Matrix" in chart_select:
        st.markdown("#### Figure 3: Confusion Matrices")
        show_plot("confusion_matrix.png", "Figure 3: Confusion matrix comparisons on the 20% holdout test set (3,947 instances)")

    if chart_select == "All Figures" or "ROC Curves" in chart_select:
        st.markdown("#### Figure 4: ROC Curves")
        show_plot("roc_curve.png", "Figure 4: Receiver Operating Characteristic curves comparing LR, SVM, and RF")

    if chart_select == "All Figures" or "Model Comparison" in chart_select:
        st.markdown("#### Figure 5: Model Comparison Bar Chart")
        show_plot("model_comparison.png", "Figure 5: Multi-metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)")

with tab_about:
    st.subheader("📋 Academic Project Information & CO5 Outcome")
    st.markdown("""
    ### 🎯 Course Outcome Alignment
    **CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures.**
    
    - **Real-Life Problem:** Energy consumption spikes in residential environments lead to grid instability, higher peak-tariff electricity costs, and unnecessary carbon emissions.
    - **Machine Learning Solution:** A machine learning classification pipeline was developed comparing **Logistic Regression** (linear baseline), **Support Vector Machine (Linear SVM)**, and **Random Forest**.
    - **Evaluation Measures:** 
        - 5-Fold Stratified Cross-Validation on the training partition (avoiding test-set leakage).
        - Test set evaluation on 3,947 unseen observations across **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
    
    ### 📊 Benchmark Summary:
    | Model | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Logistic Regression** | **73.99%** | **73.88%** | **71.20%** | **71.64%** | **71.42%** | **0.8073** |
    | **Support Vector Machine** | 73.72% | 73.80% | 70.92% | 72.02% | 71.47% | 0.8067 |
    | **Random Forest** | **88.41%** | **89.41%** | **87.83%** | **89.10%** | **88.46%** | **0.9562** |
    """)
