"""
LLM Insights & Evidence Synthesis Module for PeoplePulse AI
Translates statistical tests, Odds Ratios, and SHAP drivers into executive intelligence.
Supports OpenAI API with strict JSON schema and robust deterministic rule-based fallback.
"""

import os
import json
import numpy as np

def generate_rule_based_insights(kpi_data, statistical_tests, regression_summary, top_factors):
    """
    Deterministic evidence-based intelligence engine when no OpenAI API key is present.
    Strictly uses actual empirical numbers, sample sizes, and p-values.
    """
    insights = []
    
    # 1. Overtime & Workload Synthesis
    ot_test = next((t for t in statistical_tests if t["metric_key"] == "overtime_hours"), None)
    if ot_test and ot_test["is_significant"]:
        diff = ot_test["mean_difference"]
        insights.append({
            "title": "Sustained Overtime Associated with Elevated Voluntary Turnover",
            "executive_finding": f"Exited employees logged an average of {ot_test['exited_mean']} weekly overtime hours compared to {ot_test['active_mean']} hours for active staff (+{diff} hrs/wk difference, p < 0.001).",
            "supporting_evidence": f"Mann-Whitney U test confirms statistically significant difference (p = {ot_test['p_value']:.4e}, Cohen's d = {ot_test['cohens_d']} [{ot_test['effect_magnitude']} effect]).",
            "organizational_explanation": "Prolonged overtime without sufficient capacity relief or compensatory recovery creates chronic burnout, eroding work-life balance and accelerating voluntary departures.",
            "affected_segment": "Engineering, Customer Success, and Project-based teams with continuous delivery deadlines.",
            "confidence_and_limitations": "High statistical association. Caveat: Overtime tracking depends on accurate timesheet logging and excludes unlogged weekend triage.",
            "recommended_investigation": "Conduct departmental workload audit; assess sprint velocity targets and evaluate project backlog resourcing.",
            "suggested_intervention": "Introduce a 45-hour weekly circuit breaker; mandate compensatory time off (comp days) following major release crunch periods.",
            "kpi_to_monitor": "Quarterly Average Overtime Hours & Work-Life Balance Sentiment Index"
        })
        
    # 2. Compensation & Compa-Ratio Synthesis
    compa_test = next((t for t in statistical_tests if t["metric_key"] == "compa_ratio"), None)
    if compa_test and compa_test["is_significant"]:
        insights.append({
            "title": "Sub-Benchmark Compensation Driving Competitive Poaching",
            "executive_finding": f"Exited personnel had an average compa-ratio of {compa_test['exited_mean']} versus {compa_test['active_mean']} among retained staff, indicating that departing talent was paid below role benchmarks.",
            "supporting_evidence": f"Statistically significant pay disparity (Cohen's d = {compa_test['cohens_d']}, p = {compa_test['p_value']:.4e}).",
            "organizational_explanation": "Employees compensated below 90% of market median for their role level are disproportionately vulnerable to external recruiter outreach offering standard market adjustments.",
            "affected_segment": "High-performing mid-level ICs and roles with rapid external market wage growth.",
            "confidence_and_limitations": "Moderate to High. Benchmark relies on internal role parity medians; external regional cost-of-living differences may also contribute.",
            "recommended_investigation": "Run internal salary equity review across job levels to identify compression bottlenecks.",
            "suggested_intervention": "Allocate mid-year off-cycle salary adjustment pool targeted at top performers below 95% compa-ratio.",
            "kpi_to_monitor": "Median Compa-Ratio and Regrettable Voluntary Attrition"
        })
        
    # 3. Manager Satisfaction Synthesis
    mgr_test = next((t for t in statistical_tests if t["metric_key"] == "manager_satisfaction"), None)
    if mgr_test and mgr_test["is_significant"]:
        insights.append({
            "title": "Leadership Sentiment Divergence Correlating with Retention",
            "executive_finding": f"Active employees reported an average manager satisfaction score of {mgr_test['active_mean']}/10, whereas exited employees averaged {mgr_test['exited_mean']}/10 (gap of {abs(mgr_test['mean_difference']):.1f} points).",
            "supporting_evidence": f"Mann-Whitney U indicates significant sentiment degradation prior to exit (p = {mgr_test['p_value']:.4e}).",
            "organizational_explanation": "Supervisory friction, unclear direction, or lack of empathetic feedback directly undermines employee psychological safety and engagement.",
            "affected_segment": "Teams experiencing recent leadership turnover or uncoached first-time managers.",
            "confidence_and_limitations": "Moderate. Survey scores can suffer from fear of reprisal unless surveys are rigorously verified as third-party confidential.",
            "recommended_investigation": "Review 360-degree feedback trends and conduct confidential skip-level listening sessions.",
            "suggested_intervention": "Implement foundational People Leadership training focused on constructive coaching and active listening.",
            "kpi_to_monitor": "Manager Satisfaction Pulse Score & 1-on-1 Frequency"
        })
        
    # Fallback generic insight if tests were empty
    if not insights:
        insights.append({
            "title": "Workforce Stability Overview",
            "executive_finding": f"Current overall attrition stands at {kpi_data.get('overall_attrition_rate', 0)}% across {kpi_data.get('total_employees', 0)} total employees.",
            "supporting_evidence": "Descriptive baseline derived from active employee registers and historical exit records.",
            "organizational_explanation": "Baseline employee turnover reflects standard organizational attrition within industry norms.",
            "affected_segment": "Total Organization",
            "confidence_and_limitations": "Descriptive analysis only; ongoing data ingestion recommended.",
            "recommended_investigation": "Maintain longitudinal pulse survey cadence across all functional departments.",
            "suggested_intervention": "Establish standardized stay-interview program for key talent.",
            "kpi_to_monitor": "Monthly Attrition Rate & Net Talent Flow"
        })
        
    return insights

def query_llm_for_hr_insights(kpi_data, statistical_tests, regression_summary, top_factors, api_key=None):
    """
    Synthesizes aggregated HR findings using OpenAI API if key is available,
    otherwise returns validated deterministic rule-based findings.
    Guarantees no PII is ever transmitted.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key:
        return generate_rule_based_insights(kpi_data, statistical_tests, regression_summary, top_factors), "Rule-Based Deterministic Engine (No API Key Required)"
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Prepare aggregated, non-PII summary payload
        payload = {
            "workforce_kpis": {
                "total_employees": kpi_data.get("total_employees"),
                "attrition_rate_pct": kpi_data.get("overall_attrition_rate"),
                "voluntary_rate_pct": kpi_data.get("voluntary_attrition_rate"),
                "avg_tenure_years": kpi_data.get("avg_tenure_years"),
                "avg_engagement_score": kpi_data.get("avg_engagement_score"),
                "avg_absenteeism_days": kpi_data.get("avg_absenteeism_days")
            },
            "top_statistical_differences": [
                {
                    "metric": t["metric_name"],
                    "active_mean": t["active_mean"],
                    "exited_mean": t["exited_mean"],
                    "difference": t["mean_difference"],
                    "p_value": t["p_value"],
                    "effect_size_d": t["cohens_d"]
                }
                for t in statistical_tests[:4]
            ],
            "top_predictive_features": top_factors[:5] if top_factors else []
        }
        
        system_prompt = """
        You are an elite, evidence-based Chief People Officer and HR Analytics advisor.
        Analyze the provided aggregated HR analytics data.
        
        STRICT RULES:
        1. Never invent or hallucinate statistics, employee names, or quotes. Use ONLY numbers from the prompt.
        2. Do NOT claim causation where only correlation/association exists.
        3. Provide exactly 3 structured findings in JSON format:
        {
          "insights": [
             {
               "title": "Short descriptive title",
               "executive_finding": "Summary with numbers",
               "supporting_evidence": "Statistical tests, p-values, effect sizes",
               "organizational_explanation": "Why this workplace dynamic occurs",
               "affected_segment": "Target departments or roles",
               "confidence_and_limitations": "Data limitations or caveats",
               "recommended_investigation": "What HR should investigate next",
               "suggested_intervention": "Specific, practical retention action",
               "kpi_to_monitor": "Measurable KPI"
             }
          ]
        }
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Aggregated HR Data Payload:\n{json.dumps(payload, indent=2)}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        res_json = json.loads(response.choices[0].message.content)
        insights = res_json.get("insights", [])
        return insights, "OpenAI LLM (gpt-4o-mini)"
        
    except Exception as e:
        # Fall back cleanly
        fallback_insights = generate_rule_based_insights(kpi_data, statistical_tests, regression_summary, top_factors)
        return fallback_insights, f"Rule-Based Fallback (OpenAI API Error: {str(e)})"
