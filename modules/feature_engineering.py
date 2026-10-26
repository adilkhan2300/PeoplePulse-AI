"""
Feature Engineering Pipeline for PeoplePulse AI
Constructs an employee-level analytical table rigorously free of target leakage.
Computes benchmarked compa-ratios, tenure, overtime intensity, and wellbeing scores.
"""

import pandas as pd
import numpy as np
from datetime import datetime

FEATURE_DICTIONARY = {
    "tenure_years": {
        "domain": "Tenure & Demographics",
        "description": "Total length of service in years calculated from hire date to observation/exit date.",
        "limitations": "Does not reflect previous relevant external industry experience."
    },
    "weekly_working_hours": {
        "domain": "Workload & Wellbeing",
        "description": "Standard plus overtime hours worked per week.",
        "limitations": "Self-reported or badge logged; may miss offline unlogged weekend work."
    },
    "overtime_hours": {
        "domain": "Workload & Wellbeing",
        "description": "Weekly hours worked in excess of standard 40-hour work week.",
        "limitations": "Exempt salaried employees may have unstandardized overtime tracking."
    },
    "overtime_intensity_ratio": {
        "domain": "Workload & Wellbeing",
        "description": "Ratio of overtime hours to standard 40-hour baseline (overtime_hours / 40).",
        "limitations": "High intensity over brief crunch periods is normal; sustained crunch indicates structural understaffing."
    },
    "absenteeism_days": {
        "domain": "Workload & Wellbeing",
        "description": "Unscheduled days absent during the reporting period.",
        "limitations": "Must not be used to infer protected medical or personal disability status."
    },
    "workload_score": {
        "domain": "Workload & Wellbeing",
        "description": "Composite operational workload index (0-100 scale).",
        "limitations": "Combines project count with task velocity; qualitative burden may vary."
    },
    "compa_ratio": {
        "domain": "Compensation & Career",
        "description": "Employee base salary divided by the median benchmark salary for their job role and level.",
        "limitations": "Depends on the accuracy and timeliness of market compensation surveys."
    },
    "salary_change_percentage": {
        "domain": "Compensation & Career",
        "description": "Percentage change received during the latest annual compensation cycle.",
        "limitations": "Excludes one-time discretionary equity grants or sign-on bonuses."
    },
    "performance_rating": {
        "domain": "Performance & Talent",
        "description": "Formal annual performance evaluation score on 1-5 scale.",
        "limitations": "Subject to manager rater bias, central tendency bias, and grading curve strictness."
    },
    "goal_completion_percentage": {
        "domain": "Performance & Talent",
        "description": "Percentage of assigned OKRs/KPIs achieved in the evaluation cycle.",
        "limitations": "Goal difficulty varies significantly across business functions."
    },
    "manager_satisfaction": {
        "domain": "Engagement & Leadership",
        "description": "Employee sentiment score regarding their direct supervisor (1-10 scale).",
        "limitations": "Anonymized survey metric; potential fear of non-confidentiality."
    },
    "work_life_balance_score": {
        "domain": "Engagement & Leadership",
        "description": "Self-reported balance between professional workload and personal life (1-10 scale).",
        "limitations": "Subjective perception that fluctuates across business cycles."
    },
    "engagement_score": {
        "domain": "Engagement & Leadership",
        "description": "Comprehensive composite employee engagement pulse score (0-100 scale).",
        "limitations": "Survey snapshot; response fatigue can skew results."
    },
    "training_hours": {
        "domain": "Learning & Development",
        "description": "Completed structured learning and development hours in past 12 months.",
        "limitations": "Measures seat time rather than skill acquisition or direct role impact."
    }
}

def engineer_features(df_merged, observation_date="2025-12-31"):
    """
    Transforms merged raw HR datasets into a clean analytical modeling dataframe.
    Rigorously avoids target leakage by keeping exit dates/reasons out of features.
    """
    if df_merged is None or df_merged.empty:
        return None, []
        
    df = df_merged.copy()
    obs_dt = pd.to_datetime(observation_date)
    
    # 1. Target Label Engineering
    status_col = "employment_status"
    if status_col in df.columns:
        df["attrition_label"] = df[status_col].apply(
            lambda x: 1 if str(x).strip().lower() in ["terminated", "exited", "resigned", "1", "yes"] else 0
        )
    else:
        df["attrition_label"] = 0
        
    # Voluntary flag for descriptive breakdown
    if "termination_type" in df.columns:
        df["is_voluntary_exit"] = df["termination_type"].apply(
            lambda x: 1 if str(x).strip().lower() == "voluntary" else 0
        )
    else:
        df["is_voluntary_exit"] = df["attrition_label"]
        
    # 2. Tenure Calculation
    if "hire_date" in df.columns:
        df["hire_dt"] = pd.to_datetime(df["hire_date"], errors="coerce")
        # For terminated employees with termination_date, tenure is until termination_date
        # For active employees, tenure is until observation_date
        term_dt = pd.to_datetime(df.get("termination_date"), errors="coerce")
        end_dt = term_dt.fillna(obs_dt)
        # Ensure end_dt >= hire_dt
        days = (end_dt - df["hire_dt"]).dt.days.clip(lower=1)
        df["tenure_years"] = np.round(days / 365.25, 2)
        df["tenure_months"] = np.round(days / 30.44, 1)
        df["tenure_cohort"] = pd.cut(
            df["tenure_years"],
            bins=[-np.inf, 1.0, 2.0, 4.0, np.inf],
            labels=["< 1 Year (Early)", "1 - 2 Years", "2 - 4 Years", "4+ Years (Tenured)"]
        )
    else:
        df["tenure_years"] = 2.0
        df["tenure_months"] = 24.0
        df["tenure_cohort"] = "1 - 2 Years"

    # 3. Workload & Wellbeing Features
    if "weekly_working_hours" in df.columns:
        df["weekly_working_hours"] = pd.to_numeric(df["weekly_working_hours"], errors="coerce").fillna(40.0)
    else:
        df["weekly_working_hours"] = 40.0
        
    if "overtime_hours" in df.columns:
        df["overtime_hours"] = pd.to_numeric(df["overtime_hours"], errors="coerce").fillna(0.0)
    else:
        df["overtime_hours"] = (df["weekly_working_hours"] - 40.0).clip(lower=0)
        
    df["overtime_intensity_ratio"] = np.round(df["overtime_hours"] / 40.0, 3)
    df["is_high_overtime"] = (df["overtime_hours"] >= 8.0).astype(int)
    
    if "absenteeism_days" in df.columns:
        df["absenteeism_days"] = pd.to_numeric(df["absenteeism_days"], errors="coerce").fillna(0.0)
    else:
        df["absenteeism_days"] = 0.0
        
    if "workload_score" in df.columns:
        df["workload_score"] = pd.to_numeric(df["workload_score"], errors="coerce").fillna(50.0)
    else:
        df["workload_score"] = 50.0
        
    if "leave_days" in df.columns:
        df["leave_days"] = pd.to_numeric(df["leave_days"], errors="coerce").fillna(10.0)
    else:
        df["leave_days"] = 10.0

    # 4. Compensation & Compa-Ratio Feature
    if "base_salary" in df.columns:
        df["base_salary"] = pd.to_numeric(df["base_salary"], errors="coerce")
        # Calculate benchmark median salary by job role and level
        role_col = "job_role" if "job_role" in df.columns else "department"
        median_salaries = df.groupby(role_col)["base_salary"].transform("median")
        df["role_benchmark_salary"] = median_salaries.fillna(df["base_salary"].median())
        df["compa_ratio"] = np.round(df["base_salary"] / df["role_benchmark_salary"].replace(0, 1), 3)
        df["is_underpaid"] = (df["compa_ratio"] < 0.90).astype(int)
    else:
        df["base_salary"] = 75000.0
        df["compa_ratio"] = 1.0
        df["is_underpaid"] = 0
        
    if "salary_change_percentage" in df.columns:
        df["salary_change_percentage"] = pd.to_numeric(df["salary_change_percentage"], errors="coerce").fillna(3.0)
    else:
        df["salary_change_percentage"] = 3.0

    # 5. Performance & Engagement Features
    if "performance_rating" in df.columns:
        df["performance_rating"] = pd.to_numeric(df["performance_rating"], errors="coerce").fillna(3.0)
    else:
        df["performance_rating"] = 3.0
        
    if "goal_completion_percentage" in df.columns:
        df["goal_completion_percentage"] = pd.to_numeric(df["goal_completion_percentage"], errors="coerce").fillna(75.0)
    else:
        df["goal_completion_percentage"] = 75.0
        
    if "engagement_score" in df.columns:
        df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce").fillna(65.0)
    else:
        df["engagement_score"] = 65.0
        
    if "manager_satisfaction" in df.columns:
        df["manager_satisfaction"] = pd.to_numeric(df["manager_satisfaction"], errors="coerce").fillna(7.0)
    else:
        df["manager_satisfaction"] = 7.0
        
    if "work_life_balance_score" in df.columns:
        df["work_life_balance_score"] = pd.to_numeric(df["work_life_balance_score"], errors="coerce").fillna(7.0)
    else:
        df["work_life_balance_score"] = 7.0
        
    if "job_satisfaction" in df.columns:
        df["job_satisfaction"] = pd.to_numeric(df["job_satisfaction"], errors="coerce").fillna(7.0)
    else:
        df["job_satisfaction"] = 7.0
        
    if "recognition_score" in df.columns:
        df["recognition_score"] = pd.to_numeric(df["recognition_score"], errors="coerce").fillna(6.5)
    else:
        df["recognition_score"] = 6.5

    # 6. Training & Growth
    if "training_hours" in df.columns:
        df["training_hours"] = pd.to_numeric(df["training_hours"], errors="coerce").fillna(0.0)
    else:
        df["training_hours"] = 12.0
        
    if "promotion_status" in df.columns:
        df["is_recently_promoted"] = df["promotion_status"].apply(
            lambda x: 1 if "promoted" in str(x).lower() else 0
        )
    else:
        df["is_recently_promoted"] = 0

    # Clean categorical string columns
    for cat_col in ["department", "job_role", "job_level", "location", "employment_type"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("Unknown").astype(str).str.strip()
        else:
            df[cat_col] = "Standard"

    features_list = [
        "tenure_years", "weekly_working_hours", "overtime_hours", "overtime_intensity_ratio",
        "absenteeism_days", "workload_score", "compa_ratio", "salary_change_percentage",
        "performance_rating", "goal_completion_percentage", "manager_satisfaction",
        "work_life_balance_score", "engagement_score", "training_hours"
    ]
    
    return df, features_list

def get_feature_glossary():
    """Returns the comprehensive feature glossary and limitations dictionary."""
    return FEATURE_DICTIONARY

