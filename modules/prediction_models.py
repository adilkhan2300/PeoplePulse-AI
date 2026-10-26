"""
Machine Learning Attrition Prediction Engine for PeoplePulse AI
Implements Logistic Regression, Random Forest, and Gradient Boosting pipelines
with rigorous preprocessing, validation metrics, calibration, and threshold tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

NUMERIC_FEATURES = [
    "tenure_years", "weekly_working_hours", "overtime_hours",
    "overtime_intensity_ratio", "absenteeism_days", "workload_score",
    "compa_ratio", "salary_change_percentage", "performance_rating",
    "goal_completion_percentage", "manager_satisfaction",
    "work_life_balance_score", "engagement_score", "training_hours"
]

CATEGORICAL_FEATURES = [
    "department", "job_level", "employment_type"
]

def build_preprocessor(numeric_cols, categorical_cols):
    """Creates a scikit-learn ColumnTransformer for clean imputing and encoding."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )
    return preprocessor

def train_and_evaluate_models(df, test_size=0.20, random_state=42):
    """
    Splits data, trains Logistic Regression, Random Forest, and Gradient Boosting,
    and returns comprehensive validation metrics, fitted pipelines, and test sets.
    """
    if df is None or "attrition_label" not in df.columns:
        return None, "Target column 'attrition_label' not found."
        
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    if len(available_num) < 3:
        return None, "Too few numeric features available for modeling."
        
    X = df[available_num + available_cat].copy()
    y = df["attrition_label"].values
    
    # Stratified split to preserve class imbalance ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor(available_num, available_cat)
    
    model_configs = {
        "Logistic Regression (Baseline)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest (Nonlinear)": RandomForestClassifier(
            n_estimators=120, max_depth=6, class_weight="balanced", random_state=random_state
        ),
        "Gradient Boosting (Ensemble)": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state
        )
    }
    
    results = {}
    
    for name, clf in model_configs.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Fit on training set only
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities on held-out test set
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
        
        results[name] = {
            "pipeline": pipeline,
            "metrics": {
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1_Score": round(f1, 3),
                "ROC_AUC": round(roc_auc, 3),
                "PR_AUC": round(pr_auc, 3),
                "Brier_Score": round(brier, 3)
            },
            "confusion_matrix": cm,
            "calibration": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist()
            },
            "y_test": y_test,
            "y_prob": y_prob
        }
        
    context = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "available_num": available_num,
        "available_cat": available_cat,
        "results": results
    }
    
    return context, None

def evaluate_custom_threshold(y_true, y_prob, threshold=0.50):
    """Evaluates classification performance at a user-specified probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "threshold": threshold,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cm": cm.tolist()
    }

def score_entire_workforce(pipeline, df, threshold=0.50):
    """Scores active employees with attrition probability and categorizes into risk tiers."""
    if pipeline is None or df is None:
        return df
        
    df_scored = df.copy()
    try:
        probs = pipeline.predict_proba(df_scored)[:, 1]
        df_scored["predicted_attrition_risk"] = np.round(probs, 3)
        
        def assign_tier(p):
            if p >= 0.60:
                return "Elevated Risk"
            elif p >= 0.35:
                return "Moderate Risk"
            else:
                return "Low Risk"
                
        df_scored["risk_tier"] = [assign_tier(p) for p in probs]
        df_scored["predicted_exit"] = (probs >= threshold).astype(int)
    except Exception as e:
        df_scored["predicted_attrition_risk"] = 0.0
        df_scored["risk_tier"] = "Uncalculated"
        df_scored["predicted_exit"] = 0
        
    return df_scored
