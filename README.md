# Smart Home Energy Consumption Prediction

A machine learning application for classifying household appliance energy consumption using environmental sensor readings, appliance usage, and time-based features.

The project implements an end-to-end machine learning workflow, including data preprocessing, feature engineering, model evaluation, and an interactive web application built with Streamlit.

## Project Overview

Household energy consumption varies with appliance usage, indoor environmental conditions, and daily activity patterns. Identifying periods of elevated consumption can help users understand energy usage patterns and explore potential load-management strategies.

This project formulates energy consumption prediction as a binary classification problem.

Given a set of household and environmental inputs, the system predicts whether appliance energy consumption is:

- **Normal / Low:** Consumption is less than or equal to 60 Wh.
- **High:** Consumption exceeds 60 Wh.

The application provides an interactive interface for entering input values, viewing model predictions, examining confidence scores, and receiving energy-management recommendations.

**Primary model:** Logistic Regression  
**Best-performing evaluated model:** Random Forest Classifier  
**Application framework:** Streamlit  
**Machine learning framework:** scikit-learn

## Key Features

- End-to-end machine learning pipeline.
- Data preprocessing and feature engineering.
- Comparison of Logistic Regression, Support Vector Machine, and Random Forest.
- Stratified five-fold cross-validation.
- Holdout test-set evaluation.
- Classification metrics and diagnostic visualizations.
- Interactive Streamlit application for model inference.
- Confidence-score presentation and energy-management recommendations.
- Reproducible training and evaluation scripts.
- Android application source code, if included in the repository.

## Application Preview

![Smart Home Energy Predictor](screenshots/app.png)

*Interactive interface for entering household and environmental parameters and viewing energy-consumption predictions.*

## Live Application

**Streamlit application:** [Open the web application](https://energy-consumption-prediction.streamlit.app/)

The link above should point to the actual deployed application. Verify that the deployment is publicly accessible before publishing this README.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Data visualization | Matplotlib, Seaborn |
| Web application | Streamlit |
| Model persistence | Pickle (`.pkl`) |
| Version control | Git and GitHub |
| Additional application | Android Studio / Kotlin, if implemented as documented |

## Dataset

The project uses the Appliances Energy Prediction dataset from the UCI Machine Learning Repository.

- **Source:** [UCI Appliances Energy Prediction Dataset](https://archive.ics.uci.edu/dataset/353/appliances+energy+prediction)
- **Dataset ID:** 353
- **Observations:** 19,735
- **Sampling interval:** 10 minutes
- **Monitoring period:** Approximately 4.5 months
- **Target variable:** `Energy_Status`

The dataset contains appliance energy consumption measurements, lighting usage, indoor temperature and humidity readings, outdoor weather conditions, and additional environmental variables.

### Feature Engineering

The project uses 11 selected predictors:

| Feature group | Features |
|---|---|
| Indoor conditions | `T1`, `RH_1`, `T2`, `RH_2` |
| Outdoor conditions | `T_out`, `RH_out`, `Press_mm_hg`, `Windspeed` |
| Appliance usage | `lights` |
| Time-based features | `hour`, `day_of_week` |

The features represent selected environmental conditions, appliance usage, and temporal patterns that may be associated with household energy consumption.

### Target Variable

The continuous `Appliances` measurement is converted into a binary classification target using a 60 Wh threshold.

| Class | Definition | Observations | Distribution |
|---|---|---:|---:|
| 0 — Normal / Low | `Appliances <= 60 Wh` | 10,744 | 54.44% |
| 1 — High | `Appliances > 60 Wh` | 8,991 | 45.56% |

This threshold is a project-specific classification rule based on the dataset's median appliance consumption, not a universal definition of high energy usage.

## Model Development and Evaluation

Three classification algorithms were evaluated:

1. **Logistic Regression:** Interpretable linear baseline model.
2. **Support Vector Machine:** Linear classification benchmark.
3. **Random Forest Classifier:** Tree-based ensemble capable of modelling nonlinear relationships.

### Evaluation Methodology

The reported experimental workflow uses:

- An 80/20 training and holdout test split.
- 15,788 training observations.
- 3,947 holdout test observations.
- Five-fold stratified cross-validation on the training partition.
- `StandardScaler` within a scikit-learn pipeline to prevent preprocessing leakage across cross-validation folds.
- Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices for evaluation.

The holdout test partition is used to assess generalization on observations not used for model fitting.

### Performance Comparison

The following figures are the reported results from the project experiments.

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 73.99% ± 0.49% | 73.88% | 71.20% | 71.64% | 71.42% | 0.8073 |
| Support Vector Machine | 73.72% ± 0.43% | 73.80% | 70.92% | 72.02% | 71.47% | 0.8067 |
| Random Forest | 88.41% ± 0.60% | 89.41% | 87.83% | 89.10% | 88.46% | 0.9562 |

*Precision, recall, and F1-score refer to the positive class (Class 1: High Consumption).*

### Results and Interpretation

Random Forest achieved the highest reported test accuracy and ROC-AUC among the evaluated models. Its performance suggests that nonlinear relationships in the selected features may be important for classifying energy consumption.

Logistic Regression provides a simpler and more interpretable baseline. Its performance can be compared against more complex algorithms to assess the trade-off between model simplicity and predictive performance.

Although Random Forest performed best in the reported evaluation, the deployed application's model should be identified separately according to the model actually loaded by the application.

### Confusion Matrix Analysis

The reported confusion matrices contain the following results.

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 1,628 | 521 | 510 | 1,288 |
| Random Forest | 1,927 | 222 | 196 | 1,602 |

For this classification task:

- **True Positive:** A high-consumption observation correctly classified as high.
- **True Negative:** A normal or low-consumption observation correctly classified as normal or low.
- **False Positive:** A normal or low-consumption observation incorrectly classified as high.
- **False Negative:** A high-consumption observation incorrectly classified as normal or low.

These metrics help assess the types of classification errors made by each model.

## Diagnostic Visualizations

The repository includes the following evaluation plots, provided the corresponding files are present.

| Visualization | Description |
|---|---|
| Class distribution | Distribution of the two target classes |
| Correlation heatmap | Relationships between selected features and the target |
| Confusion matrix | Classification error analysis |
| ROC curves | Comparison of model discrimination performance |
| Model comparison | Comparison across multiple evaluation metrics |

### Class Distribution

![Class Distribution](plots/class_distribution.png)

### Correlation Heatmap

![Correlation Heatmap](plots/correlation_heatmap.png)

### Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

### ROC Curves

![ROC Curves](plots/roc_curve.png)

### Model Comparison

![Model Comparison](plots/model_comparison.png)

## Installation and Setup

### Prerequisites

- Python 3.9 or a compatible version supported by the pinned dependencies.
- Git, if cloning the repository.
- pip for dependency installation.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Streamlit will display the local application URL in the terminal, typically:

`http://localhost:8501`

## Model Training and Reproducibility

The `train.py` script contains the model training and evaluation workflow, assuming the repository includes the documented implementation.

Run the training script with:

```bash
python train.py
```

The script should reproduce the preprocessing, training, cross-validation, evaluation, and visualization steps implemented in the project.

For reproducible experiments, the training workflow should document the random seed, data-splitting strategy, dependency versions, model parameters, and output locations.

The saved model file, `model.pkl`, is used for inference if that is how `app.py` is implemented.

**Important:** Pickle files should only be loaded from trusted sources. Training and inference environments should use compatible versions of the libraries used to serialize the model.

## Repository Structure

```text
.
├── app.py
├── train.py
├── model.pkl
├── dataset.csv
├── requirements.txt
├── README.md
├── report.pdf
├── screenshots/
│   └── app.png
├── plots/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
└── android_app/
    └── [Android Studio project files]
```

This structure reflects the files described for the project. Update it to match the actual repository contents before publishing.

## Limitations and Future Improvements

Potential areas for further development include:

- Evaluating additional classification algorithms and hyperparameter configurations.
- Investigating feature importance and model interpretability.
- Testing the model on data collected from different households.
- Evaluating temporal and seasonal generalization.
- Improving the interface's accessibility and input validation.
- Adding automated tests for preprocessing, prediction, and application behaviour.
- Establishing continuous integration for testing and code quality.
- Documenting deployment configuration and model versioning.

The current classification output is an estimate based on the dataset and selected features. It does not directly control appliances, guarantee electricity savings, or establish that a household's electrical infrastructure is safe.

## License

This project is distributed under the MIT License only if the repository includes the corresponding `LICENSE` file.

See the [MIT License](https://opensource.org/licenses/MIT) for license terms.

## Acknowledgements

The project uses the Appliances Energy Prediction dataset made available through the UCI Machine Learning Repository.

Dataset reference: [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/353/appliances+energy+prediction)

---

**Project:** Smart Home Energy Consumption Prediction  
**Domain:** Machine Learning, Energy Analytics, and Web Application Development  
**Primary Model:** Logistic Regression  
**Best Reported Model:** Random Forest Classifier
