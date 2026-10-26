"""
Workforce Attrition Metrics & Trend Analysis Module
Calculates standard HR attrition rates, historical exit trends, tenure cohorts,
and departmental distributions.
"""

import pandas as pd
import numpy as np

def calculate_executive_kpis(df):
    """
    Computes core workforce KPIs following rigorous HR analytics methodology:
    Attrition Rate = Exits / Average Headcount * 100
    """
    if df is None or df.empty:
        return {}
        
    total_records = len(df)
    
    # Exited vs Active
    exits_mask = (df["attrition_label"] == 1) if "attrition_label" in df.columns else pd.Series([False]*total_records)
    exited_count = int(exits_mask.sum())
    active_count = total_records - exited_count
    
    # Average Headcount calculation: Start Headcount ~ Active + Exits, End Headcount = Active
    # Average Headcount = (Start + End) / 2 = Active + (Exits / 2)
    avg_headcount = active_count + (exited_count / 2.0)
    
    overall_attrition_rate = (exited_count / avg_headcount * 100.0) if avg_headcount > 0 else 0.0
    
    # Voluntary Attrition
    if "is_voluntary_exit" in df.columns:
        vol_exits = int((df["is_voluntary_exit"] == 1).sum())
        vol_attrition_rate = (vol_exits / avg_headcount * 100.0) if avg_headcount > 0 else 0.0
    else:
        vol_exits = exited_count
        vol_attrition_rate = overall_attrition_rate
        
    avg_tenure = round(df["tenure_years"].mean(), 1) if "tenure_years" in df.columns else 0.0
    avg_absenteeism = round(df["absenteeism_days"].mean(), 1) if "absenteeism_days" in df.columns else 0.0
    
    if "overtime_hours" in df.columns:
        ot_pct = round((df["overtime_hours"] > 0).mean() * 100.0, 1)
    else:
        ot_pct = 0.0
        
    avg_engagement = round(df["engagement_score"].mean(), 1) if "engagement_score" in df.columns else 0.0
    avg_performance = round(df["performance_rating"].mean(), 2) if "performance_rating" in df.columns else 0.0
    
    # Base estimated turnover cost
    cost_per_exit = 24500.0
    if "base_salary" in df.columns:
        # Standard benchmark: turnover cost is approx 33% of base salary
        cost_per_exit = float(df.loc[exits_mask, "base_salary"].mean() * 0.35) if exited_count > 0 else 25000.0
        if np.isnan(cost_per_exit):
            cost_per_exit = 25000.0
            
    total_est_cost = round(exited_count * cost_per_exit, 2)
    
    return {
        "total_employees": total_records,
        "active_employees": active_count,
        "exited_employees": exited_count,
        "overall_attrition_rate": round(overall_attrition_rate, 1),
        "voluntary_attrition_rate": round(vol_attrition_rate, 1),
        "voluntary_exits": vol_exits,
        "avg_headcount": round(avg_headcount, 1),
        "avg_tenure_years": avg_tenure,
        "avg_absenteeism_days": avg_absenteeism,
        "overtime_percentage": ot_pct,
        "avg_engagement_score": avg_engagement,
        "avg_performance_rating": avg_performance,
        "avg_cost_per_exit": round(cost_per_exit, 2),
        "total_estimated_turnover_cost": total_est_cost
    }

def get_department_breakdown(df):
    """Calculates attrition rates and employee counts across departments."""
    if df is None or "department" not in df.columns or "attrition_label" not in df.columns:
        return pd.DataFrame()
        
    grouped = df.groupby("department").agg(
        Total_Employees=("employee_id", "count"),
        Exits=("attrition_label", "sum"),
        Avg_Engagement=("engagement_score", "mean") if "engagement_score" in df.columns else ("employee_id", "count"),
        Avg_Overtime=("overtime_hours", "mean") if "overtime_hours" in df.columns else ("employee_id", "count"),
        Avg_Compa=("compa_ratio", "mean") if "compa_ratio" in df.columns else ("employee_id", "count")
    ).reset_index()
    
    grouped["Active"] = grouped["Total_Employees"] - grouped["Exits"]
    grouped["Avg_Headcount"] = grouped["Active"] + (grouped["Exits"] / 2.0)
    grouped["Attrition_Rate_%"] = np.round((grouped["Exits"] / grouped["Avg_Headcount"].replace(0, 1)) * 100, 1)
    grouped["Avg_Engagement"] = np.round(grouped["Avg_Engagement"], 1)
    grouped["Avg_Overtime"] = np.round(grouped["Avg_Overtime"], 1)
    grouped["Avg_Compa"] = np.round(grouped["Avg_Compa"], 2)
    
    return grouped.sort_values(by="Attrition_Rate_%", ascending=False)

def get_monthly_attrition_trends(df):
    """Aggregates exits by month to uncover seasonal or chronological spikes."""
    if df is None or "termination_date" not in df.columns:
        return pd.DataFrame()
        
    df_exits = df[df["attrition_label"] == 1].copy()
    df_exits["term_dt"] = pd.to_datetime(df_exits["termination_date"], errors="coerce")
    df_valid_terms = df_exits.dropna(subset=["term_dt"]).copy()
    
    if df_valid_terms.empty:
        return pd.DataFrame()
        
    df_valid_terms["YearMonth"] = df_valid_terms["term_dt"].dt.to_period("M").astype(str)
    
    trend = df_valid_terms.groupby(["YearMonth", "department"]).size().unstack(fill_value=0).reset_index()
    trend["Total_Exits"] = trend.drop(columns=["YearMonth"]).sum(axis=1)
    
    return trend.sort_values(by="YearMonth")

def get_tenure_cohort_analysis(df):
    """Examines attrition across tenure bands (survival/stay analysis)."""
    if df is None or "tenure_cohort" not in df.columns:
        return pd.DataFrame()
        
    cohorts = df.groupby("tenure_cohort", observed=True).agg(
        Total=("employee_id", "count"),
        Exits=("attrition_label", "sum"),
        Avg_Engagement=("engagement_score", "mean")
    ).reset_index()
    
    cohorts["Active"] = cohorts["Total"] - cohorts["Exits"]
    cohorts["Attrition_Rate_%"] = np.round((cohorts["Exits"] / cohorts["Total"].replace(0, 1)) * 100, 1)
    cohorts["Avg_Engagement"] = np.round(cohorts["Avg_Engagement"], 1)
    
    return cohorts
