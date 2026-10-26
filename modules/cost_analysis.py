"""
Turnover Cost & Financial Impact Modeling Module for PeoplePulse AI
Calculates comprehensive employee replacement expenses: recruitment, onboarding,
training, vacancy coverage, and productivity ramp-up transition costs.
"""

import pandas as pd
import numpy as np

def calculate_turnover_financials(
    df,
    recruitment_mult=0.18,      # % of annual base salary
    onboarding_mult=0.08,       # % of annual base salary
    training_mult=0.06,         # % of annual base salary
    avg_vacancy_days=45,        # Days to fill
    daily_vacancy_cost=250.0,   # Lost productivity / contractor cover per day
    productivity_loss_mult=0.15 # % salary lost during 90-day ramp-up
):
    """
    Computes rigorous turnover cost breakdown for exited employees based on role compensation.
    Separates direct recruitment costs from modeled vacancy and productivity drag.
    """
    if df is None or df.empty:
        return {}
        
    exited_df = df[df.get("attrition_label", 0) == 1].copy()
    num_exits = len(exited_df)
    
    if num_exits == 0:
        return {
            "num_exits": 0,
            "total_turnover_cost": 0.0,
            "avg_cost_per_exit": 0.0,
            "cost_breakdown": {},
            "dept_costs": pd.DataFrame()
        }
        
    salaries = exited_df["base_salary"].fillna(75000.0)
    
    # 1. Direct Recruitment Cost
    recruitment_total = float((salaries * recruitment_mult).sum())
    
    # 2. Onboarding Cost
    onboarding_total = float((salaries * onboarding_mult).sum())
    
    # 3. Replacement Training Cost
    training_total = float((salaries * training_mult).sum())
    
    # 4. Vacancy & Coverage Drag
    vacancy_total = float(num_exits * avg_vacancy_days * daily_vacancy_cost)
    
    # 5. Productivity Ramp-up Transition Drag (50% productivity over 90 days ~ 15% salary)
    productivity_total = float((salaries * productivity_loss_mult).sum())
    
    total_cost = recruitment_total + onboarding_total + training_total + vacancy_total + productivity_total
    avg_cost = total_cost / num_exits if num_exits > 0 else 0.0
    
    # Breakdown dictionary
    breakdown = {
        "Recruitment & Sourcing": round(recruitment_total, 2),
        "Onboarding & Integration": round(onboarding_total, 2),
        "Training & Ramp-up": round(training_total, 2),
        "Vacancy & Interim Coverage": round(vacancy_total, 2),
        "Productivity Loss (Ramp-up)": round(productivity_total, 2)
    }
    
    # Departmental distribution
    dept_costs = []
    for dept, grp in exited_df.groupby("department"):
        d_exits = len(grp)
        d_sal = grp["base_salary"].fillna(75000.0)
        d_rec = (d_sal * recruitment_mult).sum()
        d_onb = (d_sal * onboarding_mult).sum()
        d_trn = (d_sal * training_mult).sum()
        d_vac = d_exits * avg_vacancy_days * daily_vacancy_cost
        d_prd = (d_sal * productivity_loss_mult).sum()
        d_tot = d_rec + d_onb + d_trn + d_vac + d_prd
        
        dept_costs.append({
            "department": dept,
            "exits": d_exits,
            "total_cost": round(d_tot, 2),
            "avg_cost_per_exit": round(d_tot / d_exits if d_exits > 0 else 0, 2)
        })
        
    df_dept_costs = pd.DataFrame(dept_costs).sort_values(by="total_cost", ascending=False)
    
    return {
        "num_exits": num_exits,
        "total_turnover_cost": round(total_cost, 2),
        "avg_cost_per_exit": round(avg_cost, 2),
        "cost_breakdown": breakdown,
        "dept_costs": df_dept_costs
    }

def calculate_retention_roi(annual_turnover_cost, reduction_pct=0.20, initiative_investment=150000.0):
    """Calculates ROI and Net Financial Savings of a retention initiative."""
    avoided_turnover_cost = annual_turnover_cost * (reduction_pct / 100.0)
    net_savings = avoided_turnover_cost - initiative_investment
    roi_pct = (net_savings / initiative_investment * 100.0) if initiative_investment > 0 else 0.0
    
    return {
        "avoided_turnover_cost": round(avoided_turnover_cost, 2),
        "initiative_investment": round(initiative_investment, 2),
        "net_savings": round(net_savings, 2),
        "roi_pct": round(roi_pct, 1)
    }
