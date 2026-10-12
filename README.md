# Smart Energy Consumption & Peak Demand Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![Android Studio](https://img.shields.io/badge/Android%20Studio-Ready-3DDC84.svg)](https://developer.android.com/studio)
[![Course Outcome](https://img.shields.io/badge/CO%20Mapping-CO5-success.svg)](#-mapping-to-course-outcome-co5)

An end-to-end Machine Learning application, interactive Streamlit web dashboard, and native Android application designed to predict and detect **High Peak Energy Consumption Alerts** in smart buildings and households.

---

## 🌐 Live Streamlit Application

- **Streamlit App Link:** [Run Local App / Streamlit Cloud Deployment](http://localhost:8501)
- **Launch via Terminal:**
  ```bash
  streamlit run app.py
  ```

### 🖥️ Application Interface
![Streamlit App Interface](screenshots/app.png)

---

## 📊 Model Performance Comparison

In strict accordance with academic rigorous standards, model selection was governed **exclusively by 5-fold stratified cross-validation on the training set (1,200 samples)**, followed by final evaluation on the holdout test set (300 samples).

### Performance Metrics Table

| Model Architecture | Evaluation Phase | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **5-Fold CV (Train)** | **0.9325 ± 0.018** | **0.9327 ± 0.024** | **0.8801 ± 0.028** | **0.9056 ± 0.026** | **0.9880 ± 0.005** |
| | **Holdout Test Set** | **96.67%** | **97.20%** | **93.69%** | **95.41%** | **0.9898** |
| **Support Vector Machine (RBF)** | 5-Fold CV (Train) | 0.9300 ± 0.017 | 0.9415 ± 0.026 | 0.8642 ± 0.038 | 0.9007 ± 0.025 | 0.9860 ± 0.005 |
| | Holdout Test Set | 95.33% | 98.02% | 89.19% | 93.40% | 0.9900 |
| **Random Forest Classifier** | 5-Fold CV (Train) | 0.8950 ± 0.019 | 0.9003 ± 0.035 | 0.8054 ± 0.034 | 0.8496 ± 0.027 | 0.9641 ± 0.008 |
| | Holdout Test Set | 90.00% | 91.75% | 80.18% | 85.58% | 0.9736 |

### Holdout Test Confusion Matrix Breakdown (300 Samples)

| Model Name | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **186** | **3** | **7** | **104** | **3.3% (10/300)** |
| Support Vector Machine | 187 | 2 | 12 | 99 | 4.7% (14/300) |
| Random Forest | 181 | 8 | 22 | 89 | 10.0% (30/300) |

> **Critical Metric Insight:** In energy grid management, **Recall (Sensitivity)** is paramount. A missed peak event (False Negative) can cause transformer overloads and heavy utility demand charges, whereas a false alarm (False Positive) triggers only minor precautionary conservation. Logistic Regression achieved the highest Recall (**93.69%**, missing only 7 peaks out of 111).

---

## 📁 Dataset Details

- **Dataset Name:** Smart Building & Household Energy Consumption Benchmark
- **Source:** Grounded in physics-based building telemetry benchmarks adapted from the **UCI Appliances Energy Prediction** and **UCI Building Energy Efficiency (ENB2012)** datasets.
- **Number of Records:** **1,500 observations**
- **Number of Features:** **10 predictors** (7 numerical, 3 categorical) + 1 continuous ground measure + 1 binary target
- **Target Variable (`High_Consumption`):**
  - **Class 0 (Normal / Efficient Usage):** 947 samples (**63.13%**)
  - **Class 1 (High Peak Alert - Net Load ≥ 22.0 kW):** 553 samples (**36.87%**)

### Feature Dictionary

| Feature Name | Type | Range / Categories | Description |
| :--- | :---: | :---: | :--- |
| `Square_Footage` | float | 800 – 4500 sq ft | Total conditioned floor area dictating thermal envelope volume. |
| `Occupancy` | int | 1 – 8 persons | Number of building occupants contributing internal heat & plug loads. |
| `Outdoor_Temperature` | float | -5.0 – 38.0 °C | Ambient outside temperature driving conduction & HVAC heating/cooling. |
| `Humidity` | float | 20.0 – 85.0 % | Ambient relative humidity affecting air conditioning latent enthalpy. |
| `HVAC_Usage_Level` | str | Low, Medium, High | Operational HVAC intensity setting. |
| `Appliance_Load_kW` | float | 0.5 – 11.5 kW | Aggregate active draw of domestic and commercial appliances. |
| `Lighting_Load_kW` | float | 0.2 – 3.2 kW | Total illumination electrical draw. |
| `Solar_Generation_kW` | float | 0.0 – 7.0 kW | On-site rooftop photovoltaic active power generation offset. |
| `Time_of_Day` | str | Morning, Afternoon, Evening, Night | Diurnal tariff period (evening represents maximum utility stress). |
| `Day_Type` | str | Weekday, Weekend | Behavioral scheduling pattern. |
| `High_Consumption` | int | 0 or 1 | **Classification Target:** 1 if Net Load ≥ 22.0 kW (Peak Alert), else 0. |

---

## 📈 Visualizations & Graphical Analysis

### 1. Model Evaluation Comparison & Confusion Matrix
| Metric Comparison Across Algorithms | Test Confusion Matrix (Winner) |
| :---: | :---: |
| ![Model Comparison](plots/model_comparison.png) | ![Confusion Matrix](plots/confusion_matrix.png) |

### 2. Multi-Model ROC Curves & Feature Importance Ranking
| Receiver Operating Characteristic (ROC) | Random Forest Feature Importance |
| :---: | :---: |
| ![ROC Curves](plots/roc_curves.png) | ![Feature Importance](plots/feature_importance.png) |

### 3. Class Distribution & Correlation Matrix
| Target Class Distribution | Feature Correlation Heatmap |
| :---: | :---: |
| ![Class Distribution](plots/class_distribution.png) | ![Correlation Heatmap](plots/correlation_heatmap.png) |

---

## 📱 Android Studio Native Application

A full native **Android Studio Project** is included in the [`android_app/`](android_app/) directory.

- **On-Device Inference:** Uses [`EnergyPredictor.kt`](android_app/app/src/main/java/com/example/energyconsumption/EnergyPredictor.kt) to execute the trained StandardScaler normalization and regularized Logistic Regression weights directly on the device.
- **Latency & Reliability:** Operates with **< 1 ms latency** and **100% offline capability** (no cloud API or server required). Parity with scikit-learn is verified down to $< 10^{-8}$.
- **Modern UI:** Material 3 design with sliders, number inputs, spinners, interactive progress gauges, and dynamic recommendation cards.

### How to Run in Android Studio
1. Launch **Android Studio**.
2. Select **File > Open...** and open the folder:
   ```
   energy_consumption_ml/android_app
   ```
3. Let Gradle Sync finish (Android SDK is pre-configured in `local.properties`).
4. Select an Android Emulator or connected physical phone and click **Run (▶)** (`Shift + F10`).

👉 Detailed mobile setup guide: [README_ANDROID.md](README_ANDROID.md).

---

## 📂 Repository File Structure

```
energy_consumption_ml/
├── app.py                     # Streamlit web application with real-time inference & tips
├── model.pkl                  # Serialized champion model pipeline (Logistic Regression)
├── dataset.csv                # Complete operational dataset (1,500 records)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive project documentation
├── report.pdf                 # Publication-quality academic report (multi-page PDF)
├── screenshots/
│   └── app.png                # High-resolution screenshot of the Streamlit interface
├── android_app/               # Complete native Android Studio project
│   ├── app/                   # Android app module with Kotlin ML engine & Material UI
│   ├── build.gradle.kts       # Gradle root build configuration
│   ├── settings.gradle.kts    # Project settings
│   └── gradlew.bat            # Gradle wrapper executable
├── plots/                     # High-resolution (300 DPI) analytical evaluation figures
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── src/                       # Modular source code
│   ├── data_loader.py         # Data generation and hygiene validation
│   ├── train_and_evaluate.py  # ML pipeline, 5-fold CV, metrics & plots
│   └── run_pipeline.py        # Single-command pipeline runner
├── REPORT_Assignment2.md      # Detailed academic markdown report
└── README_ANDROID.md          # Mobile application guide
```

---

## 🎓 Mapping to Course Outcome (CO5)

> **"CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures"**

1. **Real-Life Problem:** Tackles building peak energy demand surges, enabling predictive demand-response that protects electrical grid stability and reduces carbon emissions.
2. **Appropriate Modeling:** Formulated three distinct inductive biases (linear probabilistic, kernel margin maximization, and tree ensemble bagging) and explained their theoretical alignment with energy conservation laws.
3. **Leakage Prevention:** Features scaled using `StandardScaler` inside a `Pipeline` to prevent validation/test set leakage.
4. **Performance Evaluation:** Evaluated across Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices using 5-Fold Stratified Cross-Validation.
5. **Practical Deployment:** Deployed via an interactive Streamlit dashboard (`app.py`) and on-device Android mobile application (`android_app/`).

---

## 🚀 Setup & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline
```bash
python src/run_pipeline.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Re-generate PDF Report & Mockup Screenshot
```bash
python generate_pdf_report.py
python create_app_screenshot.py
```
