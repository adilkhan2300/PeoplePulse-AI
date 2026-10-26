"""
Data Validation and Quality Assurance Module for PeoplePulse AI
Implements comprehensive schema verification, column mapping, date consistency,
range checks, orphaned key detection, and overall data health scoring.
"""

import pandas as pd
import numpy as np

def clean_column_name(col):
    """Normalize column names to lower snake_case."""
    return col.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")

def validate_schema(df, schema_name, schema_definition):
    """
    Validates a DataFrame against expected schema definition.
    Returns: status (bool), missing_required (list), missing_optional (list), col_mapping (dict)
    """
    if df is None or df.empty:
        return False, schema_definition["required"], schema_definition["optional"], {}
        
    normalized_cols = {clean_column_name(c): c for c in df.columns}
    
    missing_required = []
    for req in schema_definition["required"]:
        if clean_column_name(req) not in normalized_cols:
            missing_required.append(req)
            
    missing_optional = []
    for opt in schema_definition["optional"]:
        if clean_column_name(opt) not in normalized_cols:
            missing_optional.append(opt)
            
    status = len(missing_required) == 0
    return status, missing_required, missing_optional, normalized_cols

def inspect_dataset_quality(df, dataset_name):
    """
    Runs deep data quality checks on an individual dataset.
    Returns dictionary with detailed metrics and health score.
    """
    if df is None or df.empty:
        return {
            "dataset_name": dataset_name,
            "row_count": 0,
            "column_count": 0,
            "duplicates": 0,
            "missing_cells": 0,
            "missing_pct": 100.0,
            "health_score": 0,
            "issues": ["Dataset is empty or not provided."]
        }
        
    row_count = len(df)
    col_count = len(df.columns)
    
    # Check duplicates on employee_id if present
    id_col = next((c for c in df.columns if clean_column_name(c) == "employee_id"), None)
    dup_count = df.duplicated(subset=[id_col]).sum() if id_col else df.duplicated().sum()
    
    # Missing cells
    total_cells = row_count * col_count
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100) if total_cells > 0 else 0, 2)
    
    issues = []
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate employee identifiers detected.")
        
    # Range / sanity checks based on dataset
    clean_cols = {clean_column_name(c): c for c in df.columns}
    
    if "base_salary" in clean_cols:
        col = clean_cols["base_salary"]
        neg_sal = (pd.to_numeric(df[col], errors='coerce') <= 0).sum()
        if neg_sal > 0:
            issues.append(f"{neg_sal} records with invalid/negative base salary.")
            
    if "weekly_working_hours" in clean_cols:
        col = clean_cols["weekly_working_hours"]
        anomalous_hrs = (pd.to_numeric(df[col], errors='coerce') > 100).sum()
        if anomalous_hrs > 0:
            issues.append(f"{anomalous_hrs} records reporting >100 weekly work hours.")
            
    if "performance_rating" in clean_cols:
        col = clean_cols["performance_rating"]
        invalid_ratings = (~pd.to_numeric(df[col], errors='coerce').between(1, 5)).sum()
        if invalid_ratings > 0:
            issues.append(f"{invalid_ratings} performance ratings outside standard 1-5 scale.")
            
    if "hire_date" in clean_cols:
        col = clean_cols["hire_date"]
        invalid_dates = pd.to_datetime(df[col], errors='coerce').isna().sum()
        if invalid_dates > 0:
            issues.append(f"{invalid_dates} unparseable hire dates.")

    # Calculate health score (0-100)
    score = 100.0
    score -= min(40, missing_pct * 1.5)
    score -= min(30, (dup_count / row_count * 100) * 2) if row_count > 0 else 0
    score -= min(30, len(issues) * 5)
    health_score = max(0, int(score))
    
    return {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "column_count": col_count,
        "duplicates": int(dup_count),
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "health_score": health_score,
        "issues": issues
    }

def check_referential_integrity(primary_df, secondary_df, key="employee_id"):
    """
    Checks if all keys in secondary_df exist in primary_df, and vice-versa.
    """
    if primary_df is None or secondary_df is None or primary_df.empty or secondary_df.empty:
        return 0, 0, 100.0
        
    p_keys = set(primary_df[key].astype(str)) if key in primary_df.columns else set()
    s_keys = set(secondary_df[key].astype(str)) if key in secondary_df.columns else set()
    
    orphans_in_secondary = len(s_keys - p_keys)
    unmatched_in_primary = len(p_keys - s_keys)
    
    match_rate = round(len(s_keys & p_keys) / len(p_keys) * 100 if len(p_keys) > 0 else 0, 1)
    
    return orphans_in_secondary, unmatched_in_primary, match_rate
