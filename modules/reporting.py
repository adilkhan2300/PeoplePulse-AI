"""
Reporting and Export Module for PeoplePulse AI
Generates multi-sheet Excel reports, CSV downloads, and formal executive PDF dossiers
using ReportLab with clean typography, tables, and responsible AI disclosures.
"""

import io
import pandas as pd
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)

def export_multi_sheet_excel(kpis, dept_breakdown, model_metrics, recommendations, turnover_costs):
    """Generates an executive multi-tab Excel workbook."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Tab 1: Executive KPIs
        kpi_df = pd.DataFrame(list(kpis.items()), columns=["Metric", "Value"])
        kpi_df.to_excel(writer, sheet_name="Executive_KPIs", index=False)
        
        # Tab 2: Department Summary
        if isinstance(dept_breakdown, pd.DataFrame) and not dept_breakdown.empty:
            dept_breakdown.to_excel(writer, sheet_name="Department_Breakdown", index=False)
            
        # Tab 3: Model Metrics
        if isinstance(model_metrics, dict):
            m_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index": "Model"})
            m_df.to_excel(writer, sheet_name="Model_Validation", index=False)
            
        # Tab 4: Interventions
        if isinstance(recommendations, list) and recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_excel(writer, sheet_name="Retention_Interventions", index=False)
            
        # Tab 5: Cost Breakdown
        if isinstance(turnover_costs, dict) and "cost_breakdown" in turnover_costs:
            c_df = pd.DataFrame(list(turnover_costs["cost_breakdown"].items()), columns=["Expense Category", "Cost ($)"])
            c_df.to_excel(writer, sheet_name="Turnover_Financials", index=False)
            
    return output.getvalue()

def generate_pdf_management_report(
    kpi_data,
    statistical_tests,
    model_metrics,
    recommendations,
    turnover_costs,
    observation_window="2025 Calendar Year"
):
    """
    Constructs a formal, multi-page executive PDF intelligence report using ReportLab.
    Ensures complete privacy by excluding individual employee records.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    
    story = []
    
    # 1. Header Banner
    story.append(Paragraph("PeoplePulse AI — Strategic Attrition & Retention Intelligence", title_style))
    story.append(Paragraph(f"Executive Decision-Support Report | Observation Window: {observation_window} | Generated: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=12))
    
    # 2. Executive Summary
    story.append(Paragraph("1. Executive Workforce Summary", section_h1))
    summary_text = (
        f"This intelligence dossier analyzes organizational factors influencing employee attrition. "
        f"During the reporting period, total headcount was evaluated at <b>{kpi_data.get('total_employees', 0)} employees</b> "
        f"with <b>{kpi_data.get('exited_employees', 0)} recorded departures</b>. "
        f"The annualized workforce attrition rate stands at <b>{kpi_data.get('overall_attrition_rate', 0)}%</b> "
        f"(voluntary rate: <b>{kpi_data.get('voluntary_attrition_rate', 0)}%</b>). "
        f"Total estimated annualized turnover financial impact is modeled at <b>${kpi_data.get('total_estimated_turnover_cost', 0):,.2f}</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))
    
    # KPI Grid Table
    kpi_table_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Employees", str(kpi_data.get("total_employees", "N/A")), "Active Employees", str(kpi_data.get("active_employees", "N/A"))],
        ["Exits in Period", str(kpi_data.get("exited_employees", "N/A")), "Average Headcount", str(kpi_data.get("avg_headcount", "N/A"))],
        ["Overall Attrition Rate", f"{kpi_data.get('overall_attrition_rate', 0)}%", "Voluntary Attrition Rate", f"{kpi_data.get('voluntary_attrition_rate', 0)}%"],
        ["Average Tenure", f"{kpi_data.get('avg_tenure_years', 0)} yrs", "Overtime Population", f"{kpi_data.get('overtime_percentage', 0)}%"],
        ["Average Engagement", f"{kpi_data.get('avg_engagement_score', 0)} / 100", "Total Estimated Turnover Cost", f"${kpi_data.get('total_estimated_turnover_cost', 0):,.0f}"]
    ]
    t_kpis = Table(kpi_table_data, colWidths=[130, 130, 140, 140])
    t_kpis.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#1E293B")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_kpis)
    story.append(Spacer(1, 14))
    
    # 3. Statistical Root-Cause Findings
    story.append(Paragraph("2. Evidence-Based Root-Cause Analysis", section_h1))
    story.append(Paragraph(
        "Two-sample hypothesis tests (Mann-Whitney U and Welch's t-test) and multivariable regression "
        "reveal statistically significant disparities between active and departed employees:",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    stat_rows = [["Workplace Metric", "Active Mean", "Exited Mean", "Difference", "Effect Size (d)", "p-value"]]
    for t in statistical_tests[:5]:
        stat_rows.append([
            t["metric_name"][:28],
            str(t["active_mean"]),
            str(t["exited_mean"]),
            str(t["mean_difference"]),
            f"{t['cohens_d']} ({t['effect_magnitude']})",
            f"{t['p_value']:.4e}" if t['p_value'] < 0.001 else f"{t['p_value']:.3f}"
        ])
        
    t_stats = Table(stat_rows, colWidths=[150, 75, 75, 75, 100, 65])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 14))
    
    # 4. Machine Learning Model Evaluation
    story.append(Paragraph("3. Predictive Modeling & Machine Learning Validation", section_h1))
    story.append(Paragraph(
        "Predictive models evaluate multi-factor attrition risk on an 80/20 stratified hold-out validation set. "
        "In accordance with responsible AI practices, recall and ROC-AUC are prioritized over misleading simple accuracy:",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    if isinstance(model_metrics, dict):
        model_rows = [["Model Architecture", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]]
        for name, m in model_metrics.items():
            met = m.get("metrics", {})
            model_rows.append([
                name[:25],
                str(met.get("Accuracy", "-")),
                str(met.get("Precision", "-")),
                str(met.get("Recall", "-")),
                str(met.get("F1_Score", "-")),
                str(met.get("ROC_AUC", "-")),
                str(met.get("PR_AUC", "-"))
            ])
        t_models = Table(model_rows, colWidths=[150, 65, 65, 65, 65, 65, 65])
        t_models.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_models)
    story.append(Spacer(1, 14))
    
    # 5. Financial Turnover Cost Breakdown
    story.append(Paragraph("4. Financial Turnover Impact Modeling", section_h1))
    if isinstance(turnover_costs, dict) and "cost_breakdown" in turnover_costs:
        cost_rows = [["Turnover Expense Component", "Estimated Cost ($)", "Percentage of Total"]]
        total_c = turnover_costs.get("total_turnover_cost", 1.0)
        for cat, val in turnover_costs["cost_breakdown"].items():
            pct = round((val / total_c * 100) if total_c > 0 else 0, 1)
            cost_rows.append([cat, f"${val:,.2f}", f"{pct}%"])
            
        t_cost = Table(cost_rows, colWidths=[240, 150, 150])
        t_cost.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_cost)
    story.append(Spacer(1, 14))
    
    # 6. Strategic Retention Recommendations
    story.append(Paragraph("5. Prioritized Retention Interventions", section_h1))
    if isinstance(recommendations, list) and recommendations:
        for idx, rec in enumerate(recommendations[:3], 1):
            rec_text = (
                f"<b>{idx}. {rec.get('issue')}</b> ({rec.get('pillar')})<br/>"
                f"<b>Action:</b> {rec.get('proposed_intervention')}<br/>"
                f"<b>Target Population:</b> {rec.get('affected_population')} | <b>Lead:</b> {rec.get('responsible_function')} | <b>Timeline:</b> {rec.get('timeline')}<br/>"
                f"<b>Success Metric:</b> {rec.get('success_metric')}"
            )
            story.append(Paragraph(rec_text, body_style))
            story.append(Spacer(1, 6))
            
    # 7. Privacy, Fairness & Responsible AI Disclosure
    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Responsible AI, Privacy & Non-Causal Disclosures", section_h1))
    disclaimer_text = (
        "<b>Notice:</b> PeoplePulse AI is an organizational decision-support platform, NOT an automated employee judgment system. "
        "1) <b>No Adverse Employment Decisions:</b> Model outputs and risk probabilities must NEVER be used for disciplinary actions, termination, or compensation penalties. "
        "2) <b>Privacy Safeguards:</b> All employee personal records and names have been anonymized; individual records are excluded from executive dossiers. "
        "3) <b>Non-Causal Evidence:</b> Observed statistical associations and machine learning feature importances reflect correlation patterns, not proven causal interventions. "
        "All recommended policy adjustments require human HR governance review before implementation."
    )
    story.append(Paragraph(disclaimer_text, callout_style))
    
    doc.build(story)
    return buffer.getvalue()
