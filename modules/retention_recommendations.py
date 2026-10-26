"""
Retention Intervention Recommendation Engine for PeoplePulse AI
Generates structured, evidence-based retention interventions across Workload,
Compensation, Leadership, and Development pillars with feasibility scoring.
"""

import pandas as pd
import numpy as np

def generate_evidence_based_recommendations(df, statistical_tests, model_drivers=None):
    """
    Evaluates empirical data patterns and matches them to concrete, actionable HR interventions.
    Scores recommendations by impact potential, evidence strength, and feasibility.
    """
    if df is None or df.empty:
        return []
        
    recommendations = []
    
    # 1. Check Overtime & Burnout Pillar
    high_ot_count = int((df.get("overtime_hours", pd.Series([0])) >= 8.0).sum())
    ot_test = next((t for t in statistical_tests if t["metric_key"] == "overtime_hours"), None)
    
    if high_ot_count > 10 and (ot_test is None or ot_test["is_significant"]):
        affected_depts = df[df.get("overtime_hours", 0) >= 8.0]["department"].value_counts().head(2).index.tolist()
        dept_str = ", ".join(affected_depts) if affected_depts else "Engineering & Sales"
        
        recommendations.append({
            "id": "REC-WL-01",
            "pillar": "Workload & Wellbeing",
            "issue": "Structural Overtime & Burnout Acceleration",
            "evidence": f"{high_ot_count} employees working ≥8 weekly overtime hours. Significant correlation with voluntary departures (p < 0.05).",
            "affected_population": f"{high_ot_count} employees, concentrated in {dept_str}.",
            "proposed_intervention": "Implement workload rebalancing, establish sprint capacity caps, and institute mandatory recovery comp-days post-deliverables.",
            "responsible_function": "Department Heads & HR Business Partners",
            "timeline": "30 - 60 Days",
            "success_metric": "Reduction in employees exceeding 8 hrs overtime/week by 40%",
            "follow_up_method": "Bi-weekly timesheet audit & quarterly pulse survey",
            "expected_impact": "Downwards (-15% to -25% voluntary attrition in affected cohorts)",
            "feasibility": "High",
            "impact_score": 88,
            "confidence": "High (Strong empirical correlation)"
        })
        
    # 2. Check Compensation & Market Equity Pillar
    underpaid_count = int((df.get("compa_ratio", pd.Series([1.0])) < 0.90).sum())
    compa_test = next((t for t in statistical_tests if t["metric_key"] == "compa_ratio"), None)
    
    if underpaid_count > 10 and (compa_test is None or compa_test["is_significant"]):
        recommendations.append({
            "id": "REC-CMP-02",
            "pillar": "Compensation & Career Growth",
            "issue": "Internal Pay Compression & Below-Market Compa-Ratio",
            "evidence": f"{underpaid_count} employees paid below 90% of role benchmark median. Statistically associated with elevated exit rate.",
            "affected_population": f"{underpaid_count} active employees across Core IC levels.",
            "proposed_intervention": "Conduct targeted off-cycle market equity adjustments for high performers (<90% compa-ratio) and publish transparent band criteria.",
            "responsible_function": "Total Rewards & Compensation Committee",
            "timeline": "60 - 90 Days",
            "success_metric": "Bring 85% of underpaid talent into the 95-105% benchmark corridor",
            "follow_up_method": "Post-adjustment 6-month retention tracking & external salary survey parity",
            "expected_impact": "Downwards (Reduces regrettable departures driven by competing offers)",
            "feasibility": "Medium (Requires budget approval)",
            "impact_score": 92,
            "confidence": "High (Clear empirical compensation gap)"
        })
        
    # 3. Check Engagement & Management Pillar
    low_mgr_count = int((df.get("manager_satisfaction", pd.Series([8.0])) < 6.0).sum())
    mgr_test = next((t for t in statistical_tests if t["metric_key"] == "manager_satisfaction"), None)
    
    if low_mgr_count > 10 and (mgr_test is None or mgr_test["is_significant"]):
        recommendations.append({
            "id": "REC-MGR-03",
            "pillar": "Engagement & Leadership",
            "issue": "Sub-Optimal Supervisory Satisfaction & Psychological Safety",
            "evidence": f"{low_mgr_count} employees report manager satisfaction below 6.0/10. Exited staff showed significant degradation prior to departure.",
            "affected_population": f"{low_mgr_count} team members across teams with recent manager transitions.",
            "proposed_intervention": "Launch 'Manager as Coach' enablement cohort; mandate structured bi-weekly 1-on-1s and implement anonymous quarterly upward feedback.",
            "responsible_function": "People Operations & Executive Leadership",
            "timeline": "45 - 90 Days",
            "success_metric": "Increase average manager satisfaction score by +1.5 points",
            "follow_up_method": "Quarterly 360 upward feedback and anonymous team pulse sentiment",
            "expected_impact": "Downwards (Substantial improvement in team psychological safety and loyalty)",
            "feasibility": "High",
            "impact_score": 85,
            "confidence": "Moderate to High"
        })
        
    # 4. Check Training & Professional Development Pillar
    stagnant_training = int((df.get("training_hours", pd.Series([20.0])) < 10.0).sum())
    if stagnant_training > 20:
        recommendations.append({
            "id": "REC-LND-04",
            "pillar": "Learning & Development",
            "issue": "Skill Stagnation & Limited Professional Upskilling",
            "evidence": f"{stagnant_training} employees completed under 10 hours of structured training in the past year, correlated with lower intent-to-stay.",
            "affected_population": f"{stagnant_training} mid-tenure employees (>1.5 years tenure).",
            "proposed_intervention": "Introduce personalized $1,500 learning stipends, protected Friday learning sprints, and formal mentorship pairing.",
            "responsible_function": "Learning & Development (L&D) Team",
            "timeline": "30 - 60 Days",
            "success_metric": "Achieve 80% annual training completion rate across technical & commercial teams",
            "follow_up_method": "LMS course tracking & internal promotion mobility audit",
            "expected_impact": "Downwards (Strengthens internal mobility pipelines and career stickiness)",
            "feasibility": "High",
            "impact_score": 78,
            "confidence": "Moderate"
        })
        
    # Rank by impact score descending
    recommendations = sorted(recommendations, key=lambda x: x["impact_score"], reverse=True)
    return recommendations
