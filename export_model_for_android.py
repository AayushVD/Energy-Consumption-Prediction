"""
export_model_for_android.py
Exports the trained scikit-learn Logistic Regression Pipeline parameters
into JSON and Kotlin formats for on-device Android inference.
"""

import os
import json
import joblib


def export_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'energy_model.joblib')
    assets_dir = os.path.join(base_dir, 'android_app', 'app', 'src', 'main', 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    print(f"[INFO] Loading model bundle from: {model_path}")
    bundle = joblib.load(model_path)
    pipe = bundle['pipeline']
    pre = pipe.named_steps['preprocessor']
    clf = pipe.named_steps['classifier']

    scaler = pre.named_transformers_['num']
    ohe = pre.named_transformers_['cat']

    num_features = bundle['numerical_features']
    cat_features = bundle['categorical_features']

    model_data = {
        "model_name": bundle['model_name'],
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "scaler": {
            "mean": list(scaler.mean_),
            "scale": list(scaler.scale_)
        },
        "one_hot_categories": {
            cat_features[i]: list(ohe.categories_[i]) for i in range(len(cat_features))
        },
        "encoded_cat_names": list(ohe.get_feature_names_out(cat_features)),
        "coefficients": list(clf.coef_[0]),
        "intercept": float(clf.intercept_[0]),
        "threshold": 0.50
    }

    json_path = os.path.join(assets_dir, 'model_params.json')
    with open(json_path, 'w') as f:
        json.dump(model_data, f, indent=4)
    print(f"[SUCCESS] Exported JSON model parameters to: {json_path}")

    return model_data


if __name__ == '__main__':
    export_model()
