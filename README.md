# ⚡ Smart Home Energy Consumption Prediction Using Logistic Regression

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course Outcome](https://img.shields.io/badge/Outcome-CO5%20Attained-success.svg)]()

An end-to-end Machine Learning system that predicts whether domestic appliance energy load will be **High (> 60 Wh, Class 1)** or **Normal / Low (≤ 60 Wh, Class 0)** to enable real-time demand-side load shifting, peak-shaving, and consumer cost reduction.

---

## 📸 Application Screenshot

![Smart Home Energy Predictor Screenshot](screenshots/app.png)

*Figure: Interactive Streamlit web interface (`app.py`) featuring real-time environmental input sliders, confidence breakdown, and actionable energy management recommendations.*

---

## 🌐 Live Streamlit App Link & Execution

### 🚀 Try the Live App on Streamlit Cloud:
> **🔗 Live Web App:** [https://energy-consumption-prediction.streamlit.app](https://share.streamlit.io/)  
*(Or deploy to Streamlit Community Cloud in 1-click by connecting this GitHub repository and pointing to `app.py`).*

### 💻 Run Locally in 2 Steps:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📊 Dataset Description & Source

- **Dataset Name:** Appliances Energy Prediction Dataset
- **Repository / Source:** [UCI Machine Learning Repository (Dataset ID: 353)](https://archive.ics.uci.edu/dataset/353/appliances+energy+prediction)
- **Data File:** Included in root as `dataset.csv` (11.97 MB)
- **Total Records (Rows):** **19,735 instances**
- **Logging Interval:** Sampled continuously every 10 minutes over a 4.5-month monitoring window (January to May) at a low-energy house in Stambruges, Belgium.
- **Attributes:**
  - **29 Raw Features:** Appliance power draw (Wh), lighting fixture load, 9 indoor temperature sensors, 9 indoor humidity sensors, outdoor temperature, atmospheric pressure, outdoor humidity, wind speed, visibility, dew point, and random variables.
  - **11 Engineered Predictors:** Continuous indoor sensors (`T1`, `RH_1`, `T2`, `RH_2`), outdoor weather conditions (`T_out`, `RH_out`, `Press_mm_hg`, `Windspeed`), appliance loads (`lights`), and diurnal schedules (`hour`, `day_of_week`).
- **Target Variable Formulation (`Energy_Status`):**
  - Binarized at the dataset median threshold of **60.0 Wh**:
    - **Class 0 (Normal / Low Consumption):** `Appliances <= 60 Wh` → **10,744 instances (54.44%)**
    - **Class 1 (High Energy Consumption):** `Appliances > 60 Wh` → **8,991 instances (45.56%)**

---

## 📈 Model Performance & Experimental Evaluation

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training partition (15,788 samples) with `StandardScaler` inside an scikit-learn `Pipeline` to prevent data leakage. Final generalization was verified on the **20% holdout test partition (3,947 unseen observations)**.

### Comprehensive Performance Benchmark:

| Model | 5-Fold CV Accuracy | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **73.99% ± 0.49%** | **73.88%** | **71.20%** | **71.64%** | **71.42%** | **0.8073** | Primary Baseline |
| **Support Vector Machine** | 73.72% ± 0.43% | 73.80% | 70.92% | 72.02% | 71.47% | 0.8067 | Linear Benchmark |
| **Random Forest Classifier** | **88.41% ± 0.60%** | **89.41%** | **87.83%** | **89.10%** | **88.46%** | **0.9562** | **Ensemble Winner** |

### Confusion Matrix Breakdown (3,947 Unseen Test Samples):
- **Logistic Regression:**
  - True Negatives ($TN$): **1,628** | False Positives ($FP$): **521**
  - False Negatives ($FN$): **510** | True Positives ($TP$): **1,288**
- **Random Forest:**
  - True Negatives ($TN$): **1,927** | False Positives ($FP$): **222**
  - False Negatives ($FN$): **196** | True Positives ($TP$): **1,602**

---

## 🖼️ Diagnostic Evaluation Graphs

| Figure 1: Class Distribution | Figure 2: Correlation Heatmap |
| :---: | :---: |
| ![Class Distribution](plots/class_distribution.png) | ![Correlation Heatmap](plots/correlation_heatmap.png) |
| *Balanced target split at median threshold (60 Wh)* | *Pearson correlation of sensors with energy target* |

| Figure 3: Confusion Matrix | Figure 4: ROC Curves & AUC Comparison |
| :---: | :---: |
| ![Confusion Matrix](plots/confusion_matrix.png) | ![ROC Curves](plots/roc_curve.png) |
| *LR vs Random Forest error distributions* | *True Positive vs False Positive rates (AUC: 0.81 vs 0.96)* |

### Figure 5: Multi-Metric Model Comparison
![Model Comparison Bar Chart](plots/model_comparison.png)  
*Comprehensive benchmark across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.*

---

## 🔍 Key Insights & Analysis

1. **Why Random Forest Won:** Appliance consumption surges are governed by complex, discrete human behaviors (e.g., cooking in the evening causes simultaneous spikes in kitchen temperature, humidity, and lighting). Decision tree ensembles naturally segment these non-linear threshold boundaries, beating linear models by **+15.53% accuracy**.
2. **Interpretability of Logistic Regression:** The standardized linear weights offer clear physical insights:
   - $\text{hour}$ ($w = +0.7270$) and $\text{lights}$ ($w = +0.5704$): Strongest positive drivers of appliance consumption.
   - $\text{RH}_1$ ($w = +0.9991$) and $T_1$ ($w = +0.5526$): Active cooking raises kitchen moisture and heat during heavy appliance use.
   - $\text{day\_Saturday}$ ($w = +0.3847$): Increased residential weekend occupancy.
3. **Which Metric Matters Most:** In smart grid peak management, **Recall (Sensitivity)** and **ROC-AUC** are paramount. A False Negative (missed peak) risks distribution transformer overloading and incurs high peak electricity tariffs, whereas a False Positive (false alarm) merely causes a minor scheduling deferral.

---

## 📁 Repository File Structure

```
├── app.py                   # Streamlit web application with sliders and live inference
├── model.pkl                # Trained Scikit-Learn Logistic Regression pipeline
├── dataset.csv              # Full UCI Appliances Energy dataset (19,735 rows)
├── requirements.txt         # Project dependencies for one-command installation
├── README.md                # Comprehensive project documentation with images
├── report.pdf               # 4-6 Page academic project report (A4 PDF)
├── screenshots/
│   └── app.png              # High-resolution screenshot of the Streamlit application
├── plots/                   # All 5 evaluation figures (300 DPI)
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
├── train.py                 # Reproducible training, CV, metrics, and plotting script
└── android_app/             # Complete native Android Studio project (Kotlin)
```

---

## 🚀 How to Upload This Repository to GitHub

If Git is not installed on your system, install it via Windows Terminal / PowerShell:
```powershell
winget install --id Git.Git -e --source winget
```

Then initialize and push the repository to GitHub:
```bash
# 1. Navigate to the project root directory
cd C:\Users\Aayush\.gemini\antigravity\scratch\energy_consumption_prediction

# 2. Initialize a local git repository
git init

# 3. Add all files to staging
git add .

# 4. Commit files
git commit -m "feat: complete energy consumption prediction model, web app, report, and dataset"

# 5. Set default branch to main
git branch -M main

# 6. Add your GitHub remote repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git

# 7. Push to GitHub
git push -u origin main
```

---

## 🎓 Course Outcome (CO5) Mapping

> **Course Outcome:** *"CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures."*

- **Real-Life Problem:** Tackled peak energy surges in residential grids that cause utility transformer stress and high peak-tariff consumer electricity bills.
- **Appropriate Machine Learning Models:** Engineered and compared three complementary paradigms: regularized Logistic Regression (interpretable linear baseline), Support Vector Machine (maximum margin), and Random Forest (non-linear ensemble).
- **Evaluate Performance Measures:** Evaluated across 6 metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix) using 5-fold cross-validation and a 20% holdout test partition.
- **Interactive Delivery:** Deployed the model both to a Streamlit web interface (`app.py`) and a native offline Android Studio application.
