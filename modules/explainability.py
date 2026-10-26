"""
Explainable AI & Feature Attribution Module for PeoplePulse AI
Provides SHAP values (TreeExplainer/LinearExplainer) and Scikit-Learn Permutation Importance
to ensure robust, explainable root-cause insights across models.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def get_transformed_feature_names(preprocessor, num_cols, cat_cols):
    """Retrieves human-readable feature names post ColumnTransformer processing."""
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_encoded_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        return list(num_cols) + cat_encoded_names
    except Exception:
        # Fallback names
        return list(num_cols) + [f"cat_{i}" for i in range(10)]

def compute_permutation_importance(pipeline, X_test, y_test, num_cols, cat_cols, n_repeats=5, random_state=42):
    """
    Computes model-agnostic Permutation Feature Importance.
    Extremely robust and directly quantifies the drop in ROC-AUC when a feature is shuffled.
    """
    try:
        result = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="roc_auc",
            n_jobs=-1
        )
        
        feature_names = list(X_test.columns)
        importances = pd.DataFrame({
            "Feature": feature_names,
            "Importance_Mean": np.round(result.importances_mean, 4),
            "Importance_Std": np.round(result.importances_std, 4)
        }).sort_values(by="Importance_Mean", ascending=False)
        
        return importances, "Permutation Importance (ROC-AUC drop)"
    except Exception as e:
        return pd.DataFrame(), f"Permutation importance failed: {str(e)}"

def compute_shap_importance(pipeline, X_train, X_test, num_cols, cat_cols, sample_size=100):
    """
    Attempts to compute SHAP values using TreeExplainer or LinearExplainer.
    Falls back gracefully to Permutation Importance if SHAP encounters compatibility issues.
    """
    try:
        import shap
        
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        
        # Transform samples
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        feature_names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)
        
        # Sample for fast interactive performance
        sub_sample = X_test_trans[:min(sample_size, len(X_test_trans))]
        
        # Select appropriate explainer
        class_name = classifier.__class__.__name__
        if "Forest" in class_name or "Gradient" in class_name:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(sub_sample)
            # Binary classification check: sometimes list [class0, class1] or 3D array
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                shap_vals = shap_values[:, :, 1]
            else:
                shap_vals = shap_values
        elif "Logistic" in class_name:
            explainer = shap.LinearExplainer(classifier, X_train_trans)
            shap_vals = explainer.shap_values(sub_sample)
        else:
            explainer = shap.Explainer(classifier, X_train_trans)
            shap_vals = explainer(sub_sample).values
            
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        
        # Handle matching dimensions
        if len(mean_abs_shap) == len(feature_names):
            df_shap = pd.DataFrame({
                "Feature": feature_names,
                "SHAP_Importance": np.round(mean_abs_shap, 4)
            }).sort_values(by="SHAP_Importance", ascending=False)
            return df_shap, "SHAP (Mean |SHAP Value|)"
        else:
            return None, "Dimension mismatch in SHAP calculation"
            
    except Exception as e:
        # Graceful fallback indicator
        return None, f"SHAP calculation unavailable: {str(e)}"
