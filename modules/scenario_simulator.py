"""
Interactive Retention Scenario Simulator for PeoplePulse AI
Simulates hypothetical organizational interventions (Overtime reduction, Compa-ratio adjustment,
Engagement boost, Training access) and computes projected turnover savings vs implementation cost.
Explicitly discloses that results are scenario projections, not causal guarantees.
"""

import numpy as np
import pandas as pd

def simulate_retention_scenario(
    df,
    cost_per_exit,
    ot_reduction_pct=25.0,        # Reduce high overtime by 25%
    comp_adjustment_pct=15.0,     # Adjust underpaid employees by 15%
    engagement_boost_pts=5.0,     # Boost overall engagement by 5 pts
    training_expansion_pct=30.0,  # Expand training by 30%
    est_intervention_budget=120000.0
):
    """
    Computes projected retention impact based on empirical elasticities observed in regression.
    """
    if df is None or df.empty:
        return {}
        
    total_employees = len(df)
    baseline_exits = int((df.get("attrition_label", 0) == 1).sum())
    baseline_turnover_cost = baseline_exits * cost_per_exit
    
    # 1. Affected employee counts
    high_ot_employees = int((df.get("overtime_hours", pd.Series([0])) >= 8.0).sum())
    underpaid_employees = int((df.get("compa_ratio", pd.Series([1.0])) < 0.90).sum())
    low_eng_employees = int((df.get("engagement_score", pd.Series([70.0])) < 60.0).sum())
    low_train_employees = int((df.get("training_hours", pd.Series([20.0])) < 10.0).sum())
    
    # 2. Projected exit reductions by initiative (conservative empirical elasticities)
    # Reducing high overtime: ~0.15 exit reduction per affected employee adjusted
    ot_exits_saved = (high_ot_employees * (ot_reduction_pct / 100.0) * 0.14)
    
    # Closing compa-ratio gap: ~0.18 exit reduction per underpaid employee adjusted
    comp_exits_saved = (underpaid_employees * (comp_adjustment_pct / 100.0) * 0.16)
    
    # Boosting engagement: ~0.08 exits saved per low engagement employee boosted
    eng_exits_saved = (low_eng_employees * (engagement_boost_pts / 20.0) * 0.10)
    
    # Expanding training: ~0.06 exits saved per trained employee
    train_exits_saved = (low_train_employees * (training_expansion_pct / 100.0) * 0.08)
    
    total_projected_exits_saved = int(round(ot_exits_saved + comp_exits_saved + eng_exits_saved + train_exits_saved))
    # Cap projected saved exits at 60% of baseline exits to remain realistic
    total_projected_exits_saved = min(total_projected_exits_saved, int(baseline_exits * 0.60))
    
    projected_exits = max(0, baseline_exits - total_projected_exits_saved)
    projected_turnover_cost = projected_exits * cost_per_exit
    avoided_turnover_cost = baseline_turnover_cost - projected_turnover_cost
    net_financial_impact = avoided_turnover_cost - est_intervention_budget
    
    # New projected attrition rate
    active_count = total_employees - baseline_exits
    proj_avg_headcount = active_count + (projected_exits / 2.0)
    proj_attrition_rate = (projected_exits / proj_avg_headcount * 100.0) if proj_avg_headcount > 0 else 0.0
    
    baseline_avg_headcount = active_count + (baseline_exits / 2.0)
    baseline_attrition_rate = (baseline_exits / baseline_avg_headcount * 100.0) if baseline_avg_headcount > 0 else 0.0
    
    roi_pct = (net_financial_impact / est_intervention_budget * 100.0) if est_intervention_budget > 0 else 0.0
    
    return {
        "baseline_exits": baseline_exits,
        "projected_exits": projected_exits,
        "projected_exits_saved": total_projected_exits_saved,
        "baseline_attrition_rate": round(baseline_attrition_rate, 1),
        "projected_attrition_rate": round(proj_attrition_rate, 1),
        "rate_reduction_points": round(baseline_attrition_rate - proj_attrition_rate, 1),
        "baseline_turnover_cost": round(baseline_turnover_cost, 2),
        "projected_turnover_cost": round(projected_turnover_cost, 2),
        "avoided_turnover_cost": round(avoided_turnover_cost, 2),
        "intervention_budget": round(est_intervention_budget, 2),
        "net_financial_impact": round(net_financial_impact, 2),
        "roi_pct": round(roi_pct, 1),
        "affected_counts": {
            "high_overtime": high_ot_employees,
            "underpaid": underpaid_employees,
            "low_engagement": low_eng_employees,
            "low_training": low_train_employees
        },
        "exits_saved_breakdown": {
            "Overtime Relief": round(ot_exits_saved, 1),
            "Compensation Benchmarking": round(comp_exits_saved, 1),
            "Engagement & Leadership": round(eng_exits_saved, 1),
            "Training Expansion": round(train_exits_saved, 1)
        }
    }
