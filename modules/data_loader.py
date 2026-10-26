"""
Data Loader Module for PeoplePulse AI
Handles CSV/Excel uploads, synthetic dataset fallbacks, dataset merging on employee_id,
and sample template generation for the 7 standard HR schemas.
"""

import os
import io
import pandas as pd
import streamlit as st

DATASET_SCHEMAS = {
    "employees": {
        "required": ["employee_id", "department", "job_role", "hire_date", "employment_status"],
        "optional": ["job_level", "location", "employment_type", "termination_date", "termination_type", "termination_reason"]
    },
    "performance": {
        "required": ["employee_id", "performance_rating"],
        "optional": ["review_date", "goal_completion_percentage", "promotion_history", "manager_feedback_score"]
    },
    "workload": {
        "required": ["employee_id", "weekly_working_hours", "overtime_hours"],
        "optional": ["reporting_period", "projects_assigned", "workload_score", "leave_days", "absenteeism_days"]
    },
    "compensation": {
        "required": ["employee_id", "base_salary"],
        "optional": ["reporting_period", "bonus", "salary_change_percentage", "compensation_band", "promotion_status"]
    },
    "engagement": {
        "required": ["employee_id", "engagement_score"],
        "optional": ["survey_date", "job_satisfaction", "manager_satisfaction", "work_life_balance_score", "recognition_score", "intent_to_stay"]
    },
    "training": {
        "required": ["employee_id", "training_hours"],
        "optional": ["training_date", "training_program", "training_completion_status", "skills_development_score"]
    },
    "recruitment_costs": {
        "required": ["department", "job_role", "recruitment_cost"],
        "optional": ["onboarding_cost", "training_cost", "average_time_to_fill_days"]
    }
}

@st.cache_data(show_spinner=False)
def load_default_datasets(data_dir="data"):
    """Loads default sample datasets from the data directory."""
    datasets = {}
    files = {
        "employees": "employees.csv",
        "performance": "performance.csv",
        "workload": "workload.csv",
        "compensation": "compensation.csv",
        "engagement": "engagement.csv",
        "training": "training.csv",
        "recruitment_costs": "recruitment_costs.csv"
    }
    
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            datasets[key] = pd.read_csv(filepath)
        else:
            datasets[key] = None
            
    return datasets

def load_uploaded_file(uploaded_file):
    """Safely loads CSV or Excel files from Streamlit uploader."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported format: {uploaded_file.name}. Please upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
        return None

def merge_hr_datasets(datasets):
    """
    Merges all employee-level datasets onto the core 'employees' table.
    Gracefully handles missing optional tables.
    """
    if "employees" not in datasets or datasets["employees"] is None or datasets["employees"].empty:
        return None, ["The primary 'employees' dataset is missing or empty."]
        
    df_merged = datasets["employees"].copy()
    logs = []
    
    # Standardize column names
    df_merged.columns = [col.strip().lower() for col in df_merged.columns]
    
    # Tables to join on employee_id
    employee_tables = ["performance", "workload", "compensation", "engagement", "training"]
    
    for table_name in employee_tables:
        if table_name in datasets and datasets[table_name] is not None and not datasets[table_name].empty:
            df_sub = datasets[table_name].copy()
            df_sub.columns = [col.strip().lower() for col in df_sub.columns]
            
            if "employee_id" in df_sub.columns:
                # Deduplicate sub table by employee_id keeping latest
                df_sub = df_sub.drop_duplicates(subset=["employee_id"], keep="last")
                
                # Check for column collision
                overlap_cols = [c for c in df_sub.columns if c in df_merged.columns and c != "employee_id"]
                if overlap_cols:
                    df_sub = df_sub.rename(columns={c: f"{c}_{table_name}" for c in overlap_cols})
                    
                df_merged = pd.merge(df_merged, df_sub, on="employee_id", how="left")
                logs.append(f"Successfully merged '{table_name}' table ({len(df_sub)} records).")
            else:
                logs.append(f"Warning: '{table_name}' dataset is missing 'employee_id' key. Skipping join.")
        else:
            logs.append(f"Notice: Optional dataset '{table_name}' was not provided.")
            
    # Join recruitment costs on department + job_role if available
    if "recruitment_costs" in datasets and datasets["recruitment_costs"] is not None and not datasets["recruitment_costs"].empty:
        df_costs = datasets["recruitment_costs"].copy()
        df_costs.columns = [col.strip().lower() for col in df_costs.columns]
        if "department" in df_costs.columns and "job_role" in df_costs.columns:
            df_costs = df_costs.drop_duplicates(subset=["department", "job_role"], keep="first")
            df_merged = pd.merge(df_merged, df_costs, on=["department", "job_role"], how="left")
            logs.append(f"Successfully joined recruitment benchmark costs by department and job role.")
            
    return df_merged, logs

def generate_csv_template(dataset_name):
    """Generates an empty sample CSV template with standard column headers."""
    if dataset_name not in DATASET_SCHEMAS:
        return ""
    cols = DATASET_SCHEMAS[dataset_name]["required"] + DATASET_SCHEMAS[dataset_name]["optional"]
    df_empty = pd.DataFrame(columns=cols)
    buffer = io.StringIO()
    df_empty.to_csv(buffer, index=False)
    return buffer.getvalue()
