"""
train.py
=============================================================================
Energy Consumption Prediction using Logistic Regression and Comparison Models
Dataset: UCI Appliances Energy Prediction Dataset
Course Outcome: CO5 - Solve real life problems using appropriate machine learning
                models and evaluate the performance measures.
=============================================================================
"""

import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

# -----------------------------------------------------------------------------
# Configuration and Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv"
DATA_FILE = os.path.join(DATA_DIR, "energydata_complete.csv")

# Set visualization style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 300})

def main():
    print("=" * 80)
    print("STEP 1 & 2: DATA LOADING AND INSPECTION")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from:\n  {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print(f"Saved dataset to: {DATA_FILE}")
    else:
        print(f"Using local dataset: {DATA_FILE}")

    df_raw = pd.read_csv(DATA_FILE)
    print(f"\n[+] Raw Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"[+] Column Names: {list(df_raw.columns)}")

    # Feature Engineering from Timestamp
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()

    # Formulate binary classification target: Energy_Status
    # Appliances > 60 Wh (median) indicates High Energy Consumption (1), else Normal/Low (0)
    median_appliances = df['Appliances'].median()
    df['Energy_Status'] = (df['Appliances'] > median_appliances).astype(int)

    print("\n[+] Target Variable Definition: 'Energy_Status'")
    print(f"    - Threshold: > {median_appliances:.1f} Wh (Median Consumption)")
    print("    - Class 0: Normal / Low Energy Consumption (<= 60 Wh)")
    print("    - Class 1: High Energy Consumption (> 60 Wh)")
    class_counts = df['Energy_Status'].value_counts()
    print(f"[+] Class Counts:\n{class_counts}")
    print(f"    Class 0 Proportion: {class_counts[0] / len(df) * 100:.2f}%")
    print(f"    Class 1 Proportion: {class_counts[1] / len(df) * 100:.2f}%")

    print("\n" + "=" * 80)
    print("STEP 3A: PREPROCESSING & DATA INTEGRITY CHECKS")
    print("=" * 80)

    # Missing values check
    missing = df.isnull().sum().sum()
    print(f"[+] Total Missing Values: {missing}")

    # Duplicates check
    duplicates = df.duplicated().sum()
    print(f"[+] Total Duplicated Rows: {duplicates}")

    # Feature selection
    num_cols = [
        'lights',       # Lighting energy consumption (Wh)
        'T1',           # Kitchen temperature (°C)
        'RH_1',         # Kitchen humidity (%)
        'T2',           # Living room temperature (°C)
        'RH_2',         # Living room humidity (%)
        'T_out',        # Ambient outside temperature (°C)
        'Press_mm_hg',  # Barometric pressure (mm Hg)
        'RH_out',       # Ambient outside humidity (%)
        'Windspeed',    # Wind speed (m/s)
        'hour'          # Hour of day (0-23)
    ]
    cat_cols = ['day_of_week']  # Categorical text column requiring encoding

    feature_cols = num_cols + cat_cols
    X = df[feature_cols]
    y = df['Energy_Status']

    print(f"[+] Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"[+] Categorical Features ({len(cat_cols)}): {cat_cols}")

    # 80/20 Stratified Split
    print("\n" + "=" * 80)
    print("STEP 3B: STRATIFIED TRAIN-TEST SPLIT (80/20)")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[+] Training Set Shape: {X_train.shape[0]} samples")
    print(f"[+] Test Set Shape:     {X_test.shape[0]} samples")
    print(f"[+] Train Class Balance: 0 -> {(y_train == 0).sum()}, 1 -> {(y_train == 1).sum()}")
    print(f"[+] Test Class Balance:  0 -> {(y_test == 0).sum()}, 1 -> {(y_test == 1).sum()}")

    # Define ColumnTransformer and Pipeline to prevent data leakage
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    print("\n" + "=" * 80)
    print("STEP 3C: ALGORITHM SELECTION AND RATIONALE")
    print("=" * 80)
    print("""
1. Logistic Regression (Baseline Model):
   - Rationale: Serves as the primary linear probabilistic baseline requested in the problem topic.
     Highly interpretable coefficients directly quantify the effect of temperature, humidity, lighting,
     and peak hours on the log-odds of high energy consumption. Fast, lightweight, and deployable to
     mobile/embedded environments.

2. Support Vector Machine (Linear SVM with probability calibration):
   - Rationale: Maximum-margin classifier that constructs an optimal separating hyperplane in
     normalized feature space. Robust against high-dimensional collinearity.

3. Random Forest Classifier (Non-linear Ensemble):
   - Rationale: Bagging ensemble of de-correlated decision trees capable of capturing complex
     non-linear interactions between weather variables, indoor comfort levels, and appliances without
     assuming linear separability.
""")

    # Instantiate pipelines
    models = {
        "Logistic Regression": Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Support Vector Machine": Pipeline([
            ('prep', preprocessor),
            ('clf', CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual='auto'), cv=3))
        ]),
        "Random Forest": Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
        ])
    }

    # 5-Fold Cross Validation
    print("=" * 80)
    print("STEP 3D: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 80)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, pipe in models.items():
        print(f"Running 5-fold cross-validation for {name}...")
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        roc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_results[name] = {
            'cv_acc_mean': scores.mean(),
            'cv_acc_std': scores.std(),
            'cv_roc_mean': roc_scores.mean(),
            'cv_roc_std': roc_scores.std()
        }
        print(f"  -> Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f}) | ROC-AUC: {roc_scores.mean():.4f}")

    print("\n" + "=" * 80)
    print("STEP 3E: TRAINING ON FULL TRAIN SET & TESTING ON TEST SET")
    print("=" * 80)

    test_metrics = {}
    fitted_models = {}
    y_probas = {}
    y_preds = {}

    for name, pipe in models.items():
        print(f"\nTraining {name} on 80% training set...")
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        y_preds[name] = y_pred
        y_probas[name] = y_proba

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        test_metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc,
            'Confusion Matrix': cm
        }

        print(f"Results for {name}:")
        print(f"  - Accuracy:  {acc:.4f}")
        print(f"  - Precision: {prec:.4f}")
        print(f"  - Recall:    {rec:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - ROC-AUC:   {roc:.4f}")
        print(f"  - Confusion Matrix:\n{cm}")

    # Print summary table
    summary_df = pd.DataFrame({
        'Model': list(test_metrics.keys()),
        '5-Fold CV Acc': [cv_results[m]['cv_acc_mean'] for m in test_metrics],
        'Test Accuracy': [test_metrics[m]['Accuracy'] for m in test_metrics],
        'Precision': [test_metrics[m]['Precision'] for m in test_metrics],
        'Recall': [test_metrics[m]['Recall'] for m in test_metrics],
        'F1-Score': [test_metrics[m]['F1-Score'] for m in test_metrics],
        'ROC-AUC': [test_metrics[m]['ROC-AUC'] for m in test_metrics]
    })
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Save models
    print("\n" + "=" * 80)
    print("SAVING TRAINED MODELS & METADATA FOR APPS")
    print("=" * 80)
    lr_pipeline_path = os.path.join(MODELS_DIR, "logistic_regression_pipeline.joblib")
    rf_pipeline_path = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
    joblib.dump(fitted_models["Logistic Regression"], lr_pipeline_path)
    joblib.dump(fitted_models["Random Forest"], rf_pipeline_path)
    print(f"[+] Saved Logistic Regression pipeline to: {lr_pipeline_path}")
    print(f"[+] Saved Random Forest pipeline to:       {rf_pipeline_path}")

    # Extract Logistic Regression parameters for Android Native Inference
    lr_clf = fitted_models["Logistic Regression"].named_steps['clf']
    scaler = fitted_models["Logistic Regression"].named_steps['prep'].named_transformers_['num']
    ohe = fitted_models["Logistic Regression"].named_steps['prep'].named_transformers_['cat']
    ohe_categories = ohe.categories_[0][1:].tolist()  # dropped first category ('Friday' or 'Monday')

    encoded_feature_names = num_cols + [f"day_{day}" for day in ohe_categories]
    weights = lr_clf.coef_[0].tolist()
    intercept = float(lr_clf.intercept_[0])

    metadata = {
        "num_features": num_cols,
        "cat_features": cat_cols,
        "encoded_features": encoded_feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ohe_categories": ohe_categories,
        "weights": weights,
        "intercept": intercept,
        "classes": ["Normal / Low Energy (<=60 Wh)", "High Energy (>60 Wh)"],
        "metrics": {
            k: {
                metric: float(v[metric]) if metric != 'Confusion Matrix' else v[metric].tolist()
                for metric in v
            } for k, v in test_metrics.items()
        }
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved model metadata and coefficients to: {metadata_path}")

    # -------------------------------------------------------------------------
    # STEP 3F: PLOTTING ALL 5 REQUIRED GRAPHS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING AND SAVING PLOTS (5 Figures)")
    print("=" * 80)

    # 1. Class Distribution Plot
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    palette = ['#2b5c8f', '#d9534f']
    sns.countplot(x='Energy_Status', data=df, palette=palette, ax=ax1, hue='Energy_Status', legend=False)
    ax1.set_title("Energy Consumption Class Distribution", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Energy Consumption Class", fontsize=12)
    ax1.set_ylabel("Count (10-minute intervals)", fontsize=12)
    ax1.set_xticklabels(["0: Normal/Low (<= 60 Wh)", "1: High (> 60 Wh)"])
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{int(height):,} ({height/len(df)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., height / 2),
                     ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plot1_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Graph 1: {plot1_path}")

    # 2. Correlation Heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    corr_cols = num_cols + ['Energy_Status']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, ax=ax2, cbar_kws={'label': 'Pearson Correlation'})
    ax2.set_title("Feature Correlation Heatmap with Target Variable", fontsize=14, fontweight='bold', pad=12)
    plot2_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Graph 2: {plot2_path}")

    # 3. Confusion Matrix (Comparing Logistic Regression and Random Forest)
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    class_labels = ["Normal (0)", "High (1)"]

    for ax, name in zip(axes3, ["Logistic Regression", "Random Forest"]):
        cm = test_metrics[name]['Confusion Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_labels, yticklabels=class_labels, cbar=False)
        ax.set_title(f"Confusion Matrix: {name}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        # Add accuracy annotation
        acc = test_metrics[name]['Accuracy']
        ax.text(0.5, -0.15, f"Test Accuracy = {acc*100:.2f}%", ha='center', transform=ax.transAxes, fontsize=11)

    plot3_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Graph 3: {plot3_path}")

    # 4. ROC Curves
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    colors = {'Logistic Regression': '#1f77b4', 'Support Vector Machine': '#ff7f0e', 'Random Forest': '#2ca02c'}
    ax4.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.50)')

    for name in models:
        fpr, tpr, _ = roc_curve(y_test, y_probas[name])
        auc_score = test_metrics[name]['ROC-AUC']
        ax4.plot(fpr, tpr, color=colors[name], lw=2.2, label=f"{name} (AUC = {auc_score:.4f})")

    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax4.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax4.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight='bold', pad=12)
    ax4.legend(loc="lower right", fontsize=10, frameon=True)
    plot4_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Graph 4: {plot4_path}")

    # 5. Model Comparison Bar Chart
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    comp_df = pd.DataFrame({
        metric: [test_metrics[m][metric] for m in models]
        for metric in metrics_to_plot
    }, index=list(models.keys()))

    comp_df.plot(kind='bar', ax=ax5, colormap='viridis', width=0.75, edgecolor='black', linewidth=0.5)
    ax5.set_title("Machine Learning Model Performance Comparison", fontsize=14, fontweight='bold', pad=12)
    ax5.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax5.set_ylim([0.5, 1.02])
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0, fontsize=11, fontweight='semibold')
    ax5.legend(loc='lower right', fontsize=10, frameon=True)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax5.patches:
        height = p.get_height()
        if height > 0.05:
            ax5.annotate(f'{height:.2f}',
                         (p.get_x() + p.get_width() / 2., height + 0.008),
                         ha='center', va='bottom', fontsize=8, rotation=90)

    plot5_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Graph 5: {plot5_path}")

    print("\n" + "=" * 80)
    print("ALL TRAINING, EVALUATION, AND PLOTTING TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
