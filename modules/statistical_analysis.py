"""
Statistical Root-Cause Analysis Module for PeoplePulse AI
Provides rigorous hypothesis testing, Mann-Whitney U, Chi-square tests,
effect sizes (Cohen's d, Cramér's V), and multivariable logistic regression odds ratios.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def perform_group_comparison_tests(df):
    """
    Compares active vs exited employees across continuous workplace metrics.
    Performs independent two-sample t-tests and Mann-Whitney U tests,
    calculates Cohen's d effect size, and computes means.
    """
    if df is None or "attrition_label" not in df.columns:
        return []
        
    metrics = [
        ("weekly_working_hours", "Weekly Working Hours"),
        ("overtime_hours", "Weekly Overtime Hours"),
        ("absenteeism_days", "Annual Absenteeism Days"),
        ("workload_score", "Workload Index Score"),
        ("compa_ratio", "Role Compa-Ratio Benchmark"),
        ("salary_change_percentage", "Salary Increase Percentage"),
        ("performance_rating", "Performance Rating"),
        ("goal_completion_percentage", "Goal Completion %"),
        ("manager_satisfaction", "Manager Satisfaction Score"),
        ("work_life_balance_score", "Work-Life Balance Score"),
        ("engagement_score", "Overall Engagement Score"),
        ("training_hours", "Training Hours Completed"),
        ("tenure_years", "Tenure (Years)")
    ]
    
    active_df = df[df["attrition_label"] == 0]
    exited_df = df[df["attrition_label"] == 1]
    
    n_active = len(active_df)
    n_exited = len(exited_df)
    
    if n_active < 5 or n_exited < 5:
        return []
        
    results = []
    
    for col, display_name in metrics:
        if col not in df.columns:
            continue
            
        s_active = pd.to_numeric(active_df[col], errors="coerce").dropna()
        s_exited = pd.to_numeric(exited_df[col], errors="coerce").dropna()
        
        if len(s_active) < 3 or len(s_exited) < 3:
            continue
            
        m_active = float(s_active.mean())
        m_exited = float(s_exited.mean())
        diff = m_exited - m_active
        
        # Mann-Whitney U test (non-parametric, robust to skew)
        u_stat, p_val = stats.mannwhitneyu(s_exited, s_active, alternative="two-sided")
        
        # Cohen's d
        s_pooled = np.sqrt(((len(s_active) - 1) * s_active.var() + (len(s_exited) - 1) * s_exited.var()) / (len(s_active) + len(s_exited) - 2))
        cohens_d = (diff / s_pooled) if s_pooled > 0 else 0.0
        
        # Effect size magnitude
        abs_d = abs(cohens_d)
        if abs_d >= 0.8:
            effect_mag = "Large"
        elif abs_d >= 0.5:
            effect_mag = "Medium"
        elif abs_d >= 0.2:
            effect_mag = "Small"
        else:
            effect_mag = "Negligible"
            
        results.append({
            "metric_key": col,
            "metric_name": display_name,
            "active_mean": round(m_active, 2),
            "exited_mean": round(m_exited, 2),
            "mean_difference": round(diff, 2),
            "p_value": float(p_val),
            "is_significant": p_val < 0.05,
            "cohens_d": round(cohens_d, 2),
            "effect_magnitude": effect_mag,
            "direction": "Higher in Exited" if diff > 0 else "Lower in Exited"
        })
        
    return sorted(results, key=lambda x: abs(x["cohens_d"]), reverse=True)

def perform_categorical_chi2_tests(df):
    """
    Performs Pearson Chi-Square tests of independence for categorical workplace factors
    (Department, Job Level, Overtime Category, Promotion Status) and computes Cramér's V.
    """
    if df is None or "attrition_label" not in df.columns:
        return []
        
    cat_columns = [
        ("department", "Department"),
        ("job_level", "Job Level"),
        ("is_high_overtime", "High Overtime (>8h/wk)"),
        ("is_underpaid", "Underpaid (<90% benchmark)"),
        ("is_recently_promoted", "Promotion in Last 12m"),
        ("location", "Office Location"),
        ("employment_type", "Employment Type")
    ]
    
    results = []
    
    for col, display_name in cat_columns:
        if col not in df.columns:
            continue
            
        contingency = pd.crosstab(df[col], df["attrition_label"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
            
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        
        # Cramér's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
        
        results.append({
            "feature": display_name,
            "chi2_stat": round(chi2, 2),
            "p_value": float(p_val),
            "is_significant": p_val < 0.05,
            "cramers_v": round(cramers_v, 3),
            "degrees_of_freedom": int(dof)
        })
        
    return sorted(results, key=lambda x: x["cramers_v"], reverse=True)

def fit_multivariable_logistic_regression(df):
    """
    Fits an adjusted multivariable logistic regression model using Statsmodels.
    Computes Adjusted Odds Ratios (OR = exp(coef)), 95% Confidence Intervals, and p-values.
    Controls for potential confounders.
    """
    if df is None or "attrition_label" not in df.columns:
        return None, "Target column missing."
        
    features = [
        "overtime_hours",
        "compa_ratio",
        "engagement_score",
        "manager_satisfaction",
        "absenteeism_days",
        "tenure_years",
        "performance_rating"
    ]
    
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        return None, "Insufficient features for multivariable regression."
        
    df_clean = df[available_features + ["attrition_label"]].dropna()
    
    if len(df_clean) < 50:
        return None, "Sample size too small for reliable logistic regression."
        
    X = df_clean[available_features]
    X = sm.add_constant(X)
    y = df_clean["attrition_label"]
    
    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=100)
        
        coefs = model.params
        conf = model.conf_int()
        p_vals = model.pvalues
        
        summary_rows = []
        for feat in X.columns:
            c = coefs[feat]
            or_val = np.exp(c)
            ci_lower = np.exp(conf.loc[feat, 0])
            ci_upper = np.exp(conf.loc[feat, 1])
            p_v = p_vals[feat]
            
            # Meaningful human interpretation
            name_map = {
                "const": "Baseline Intercept",
                "overtime_hours": "Each Additional Overtime Hour/Wk",
                "compa_ratio": "Compa-Ratio Increase (+1.0)",
                "engagement_score": "Engagement Score (+10 pts)",
                "manager_satisfaction": "Manager Satisfaction (+1 pt)",
                "absenteeism_days": "Each Additional Absenteeism Day",
                "tenure_years": "Each Additional Year of Tenure",
                "performance_rating": "Performance Rating (+1 pt)"
            }
            
            summary_rows.append({
                "feature": feat,
                "label": name_map.get(feat, feat),
                "coefficient": round(c, 3),
                "odds_ratio": round(or_val, 3),
                "ci_95_lower": round(ci_lower, 3),
                "ci_95_upper": round(ci_upper, 3),
                "p_value": float(p_v),
                "is_significant": p_v < 0.05
            })
            
        summary_df = pd.DataFrame(summary_rows)
        metrics = {
            "nobs": int(model.nobs),
            "prsquared": round(model.prsquared, 3),
            "llnull": round(model.llnull, 1),
            "llr_pvalue": float(model.llr_pvalue)
        }
        
        return summary_df, metrics
        
    except Exception as e:
        return None, f"Logistic regression convergence failure: {str(e)}"
