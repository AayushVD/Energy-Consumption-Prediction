"""
streamlit_app.py
Interactive Web Application for Energy Consumption Prediction & Smart Peak Demand Management.
Built with Streamlit.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Smart Energy Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-high {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 18px;
        border-radius: 8px;
        color: #991b1b;
        margin-top: 15px;
    }
    .alert-normal {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 18px;
        border-radius: 8px;
        color: #166534;
        margin-top: 15px;
    }
    .tip-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'energy_model.joblib')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def main():
    model_bundle = load_trained_model()

    st.markdown('<div class="main-title">⚡ Smart Energy Consumption & Peak Demand Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Machine Learning Decision Support System for Real-Time Building Energy Auditing and Demand-Response Management</div>', unsafe_allow_html=True)

    # Sidebar: Model Specs & Visualizations
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown("**Domain:** Energy Consumption Prediction")
        st.markdown("**Target:** High Peak Demand Alert (`1`) vs Normal (`0`)")
        st.markdown("**Course Outcome:** CO5 – Solve real life problems using ML models")

        if model_bundle:
            st.success(f"**Loaded Model:** {model_bundle['model_name']}")
        else:
            st.error("Model file not found. Please run `python src/run_pipeline.py` first.")

        st.markdown("---")
        st.subheader("📊 Visual Analytics Explorer")
        plot_option = st.selectbox(
            "Select Evaluation Plot to View:",
            [
                "None",
                "Model Comparison Bar Chart",
                "ROC Curves Comparison",
                "Test Confusion Matrix",
                "Target Class Distribution",
                "Feature Correlation Heatmap",
                "Feature Importance Ranking"
            ]
        )

        plot_file_map = {
            "Model Comparison Bar Chart": "model_comparison.png",
            "ROC Curves Comparison": "roc_curves.png",
            "Test Confusion Matrix": "confusion_matrix.png",
            "Target Class Distribution": "class_distribution.png",
            "Feature Correlation Heatmap": "correlation_heatmap.png",
            "Feature Importance Ranking": "feature_importance.png"
        }

        if plot_option != "None":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            p_path = os.path.join(base_dir, 'plots', plot_file_map[plot_option])
            if os.path.exists(p_path):
                st.image(p_path, caption=plot_option, use_container_width=True)
            else:
                st.warning("Plot file not found. Run pipeline first.")

    # KPI Top Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""<div class="metric-card"><h4>Dataset Size</h4><h2>1,500</h2><p>Building Observations</p></div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown("""<div class="metric-card"><h4>Cross-Validation</h4><h2>5-Fold</h2><p>Stratified Split</p></div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown("""<div class="metric-card"><h4>Model Accuracy</h4><h2>96.67%</h2><p>Holdout Test Set</p></div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown("""<div class="metric-card"><h4>ROC-AUC Score</h4><h2>0.990</h2><p>High Demand Discrimination</p></div>""", unsafe_allow_html=True)

    st.write("")
    st.subheader("🛠️ Input Building Operational Parameters")
    st.write("Adjust environmental, structural, and load parameters to evaluate whether energy consumption exceeds peak operational thresholds.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🏢 Structural & Weather")
            square_footage = st.slider("Square Footage (sq ft)", min_value=800, max_value=4500, value=2400, step=50)
            occupancy = st.slider("Occupancy (Persons)", min_value=1, max_value=8, value=4, step=1)
            outdoor_temp = st.slider("Outdoor Temperature (°C)", min_value=-5.0, max_value=38.0, value=29.0, step=0.5)
            humidity = st.slider("Relative Humidity (%)", min_value=20.0, max_value=85.0, value=55.0, step=1.0)

        with col2:
            st.markdown("#### ⚙️ Loads & HVAC")
            hvac_usage = st.selectbox("HVAC Usage Level", options=['Low', 'Medium', 'High'], index=1)
            appliance_load = st.slider("Appliance Load (kW)", min_value=0.5, max_value=11.5, value=5.2, step=0.1)
            lighting_load = st.slider("Lighting Load (kW)", min_value=0.2, max_value=3.2, value=1.5, step=0.1)
            solar_gen = st.slider("Solar Generation (kW Offset)", min_value=0.0, max_value=7.0, value=1.5, step=0.1)

        with col3:
            st.markdown("#### 🕒 Temporal Factors")
            time_of_day = st.selectbox("Time of Day", options=['Morning', 'Afternoon', 'Evening', 'Night'], index=2)
            day_type = st.selectbox("Day Type", options=['Weekday', 'Weekend'], index=0)

            st.write("")
            st.write("")
            submitted = st.form_submit_button("⚡ Predict Energy Demand Status", use_container_width=True)

    if submitted:
        if model_bundle is None:
            st.error("Model pipeline is not loaded. Please train the model first.")
            return

        pipeline = model_bundle['pipeline']

        # Construct input DataFrame matching feature columns
        input_data = pd.DataFrame([{
            'Square_Footage': float(square_footage),
            'Occupancy': int(occupancy),
            'Outdoor_Temperature': float(outdoor_temp),
            'Humidity': float(humidity),
            'HVAC_Usage_Level': hvac_usage,
            'Appliance_Load_kW': float(appliance_load),
            'Lighting_Load_kW': float(lighting_load),
            'Solar_Generation_kW': float(solar_gen),
            'Time_of_Day': time_of_day,
            'Day_Type': day_type
        }])

        # Perform inference
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        prob_high = probabilities[1] * 100
        prob_normal = probabilities[0] * 100

        st.subheader("🎯 Prediction Results & Energy Audit")
        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="alert-high">
                    <h3 style="margin-top:0;">🚨 HIGH PEAK DEMAND ALERT</h3>
                    <p><b>Status:</b> Predicted to exceed peak threshold (≥ 22.0 kW net load).</p>
                    <p><b>High Demand Probability:</b> <span style="font-size:1.3rem; font-weight:bold;">{prob_high:.1f}%</span></p>
                    <p>Immediate demand-response or load-shedding measures recommended to avoid peak utility tariff penalties.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-normal">
                    <h3 style="margin-top:0;">✅ NORMAL / EFFICIENT USAGE</h3>
                    <p><b>Status:</b> Net load is within optimal baseline parameters (&lt; 22.0 kW).</p>
                    <p><b>Normal Confidence:</b> <span style="font-size:1.3rem; font-weight:bold;">{prob_normal:.1f}%</span></p>
                    <p>Building is currently operating within green, energy-efficient operational boundaries.</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("**Peak Probability Gauge:**")
            st.progress(float(probabilities[1]))

        with res_col2:
            st.markdown("#### 💡 Actionable Energy Optimization Recommendations")
            recommendations = []

            if hvac_usage == 'High':
                recommendations.append("🌡️ **HVAC Optimization:** HVAC is currently set to 'High'. Adjusting thermostat setpoint by 1.5°C to 2°C or scheduling precooling/preheating can lower consumption by up to 18-25%.")
            if time_of_day == 'Evening' and prediction == 1:
                recommendations.append("🌙 **Peak Shifting:** Evening hours represent maximum grid stress and peak tariff rates. Shift non-urgent heavy appliance loads (dryers, dishwashers) to night (off-peak hours).")
            if appliance_load > 6.0:
                recommendations.append("🔌 **Appliance Management:** High concurrent appliance load detected ({:.1f} kW). Stagger high-draw equipment to flatten load spikes.".format(appliance_load))
            if solar_gen < 1.0 and time_of_day in ['Morning', 'Afternoon']:
                recommendations.append("☀️ **Solar Offset:** Solar offset is low during daylight hours. Inspect panel cleanliness, shade obstruction, or inverter performance.")
            if lighting_load > 2.0:
                recommendations.append("💡 **Lighting Retrofit:** Consider daylight harvesting or automated occupancy sensors to reduce lighting load from {:.1f} kW to below 1.0 kW.".format(lighting_load))

            if not recommendations:
                recommendations.append("🌿 **Optimal Operation:** All subsystem parameters are operating harmoniously. No urgent curtailment required.")

            for rec in recommendations:
                st.markdown(f"<div class='tip-box'>{rec}</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
