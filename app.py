"""
PeoplePulse AI — Strategic Attrition & Retention Intelligence Platform
Full-stack production Streamlit application for evidence-based workforce intelligence,
statistical root-cause analysis, machine learning flight risk prediction,
turnover financial modeling, scenario simulation, and retention action tracking.
"""

import os
import io
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Module imports
from modules.data_loader import (
    load_default_datasets, merge_hr_datasets, load_uploaded_file,
    generate_csv_template, DATASET_SCHEMAS
)
from modules.data_validation import validate_schema, inspect_dataset_quality
from modules.feature_engineering import engineer_features, get_feature_glossary
from modules.attrition_analysis import (
    calculate_executive_kpis, get_department_breakdown,
    get_monthly_attrition_trends, get_tenure_cohort_analysis
)
from modules.statistical_analysis import (
    perform_group_comparison_tests, perform_categorical_chi2_tests,
    fit_multivariable_logistic_regression
)
from modules.prediction_models import (
    train_and_evaluate_models, evaluate_custom_threshold,
    score_entire_workforce, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from modules.explainability import (
    compute_permutation_importance, compute_shap_importance
)
from modules.cost_analysis import (
    calculate_turnover_financials, calculate_retention_roi
)
from modules.scenario_simulator import simulate_retention_scenario
from modules.retention_recommendations import generate_evidence_based_recommendations
from modules.action_tracker import (
    init_db, add_retention_action, get_all_actions,
    update_action_status, delete_retention_action
)
from modules.llm_insights import query_llm_for_hr_insights, generate_rule_based_insights
from modules.reporting import (
    export_multi_sheet_excel, generate_pdf_management_report
)
from utils.theme import (
    apply_app_theme, render_hero, render_kpi, render_callout,
    get_plotly_layout, PRIMARY_COLOR, SECONDARY_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR
)

# Page configuration
st.set_page_config(
    page_title="PeoplePulse AI | Strategic Attrition & Retention Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek dark glassmorphism theme
apply_app_theme()

# Initialize SQLite Action Tracker Database
init_db()


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION & CACHING
# -----------------------------------------------------------------------------
def init_session_state():
    if "datasets" not in st.session_state:
        st.session_state.datasets = load_default_datasets("data")
    if "custom_mode" not in st.session_state:
        st.session_state.custom_mode = False
    if "ml_context" not in st.session_state:
        st.session_state.ml_context = None
    if "decision_threshold" not in st.session_state:
        st.session_state.decision_threshold = 0.40
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

init_session_state()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & DATA INGESTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <div style="background: linear-gradient(135deg, #3B82F6, #6366F1); padding: 8px 12px; border-radius: 8px; font-weight: 800; font-size: 18px; color: #fff;">
            👥
        </div>
        <div>
            <div style="font-weight: 800; font-size: 18px; color: #F8FAFC; letter-spacing: -0.5px;">PeoplePulse AI</div>
            <div style="font-size: 11px; color: #94A3B8;">Workforce Retention Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🗂️ Data Ingestion")
    
    data_source = st.radio(
        "Data Source Selection",
        ["Default Synthetic Benchmark (1,200 Records)", "Upload Custom HR Tables (7 Schemas)"],
        index=0 if not st.session_state.custom_mode else 1
    )
    
    if data_source.startswith("Upload"):
        st.session_state.custom_mode = True
        with st.expander("📤 Multi-Table File Upload", expanded=True):
            st.caption("Upload CSV or Excel files matching standard HR relational schemas.")
            uploaded_datasets = {}
            for schema_key in DATASET_SCHEMAS.keys():
                up_file = st.file_uploader(f"Upload '{schema_key}'", type=["csv", "xlsx", "xls"], key=f"up_{schema_key}")
                if up_file:
                    uploaded_datasets[schema_key] = load_uploaded_file(up_file)
            
            if st.button("Apply Uploaded Datasets", type="primary", use_container_width=True):
                if "employees" in uploaded_datasets and uploaded_datasets["employees"] is not None:
                    # Keep default fallbacks for missing tables
                    merged_dict = load_default_datasets("data")
                    merged_dict.update(uploaded_datasets)
                    st.session_state.datasets = merged_dict
                    st.session_state.ml_context = None # invalidate cache
                    st.success("Custom datasets applied successfully!")
                    st.rerun()
                else:
                    st.error("The core 'employees' dataset is required for joining.")
            
            st.markdown("---")
            st.caption("Download Empty CSV Template Schemas:")
            template_choice = st.selectbox("Select Schema", list(DATASET_SCHEMAS.keys()))
            csv_str = generate_csv_template(template_choice)
            st.download_button(
                f"📥 Download {template_choice}.csv Template",
                data=csv_str,
                file_name=f"{template_choice}_template.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        if st.session_state.custom_mode:
            st.session_state.custom_mode = False
            st.session_state.datasets = load_default_datasets("data")
            st.session_state.ml_context = None
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ LLM Intelligence Settings")
    api_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enables natural-language executive synthesis. Leave blank to use the deterministic statistical engine."
    )
    if api_key_input != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key_input
        
    st.markdown("---")
    st.caption("PeoplePulse AI Enterprise v2.5 | Shayam Sir IA")


# -----------------------------------------------------------------------------
# DATA PREPARATION PIPELINE
# -----------------------------------------------------------------------------
raw_datasets = st.session_state.datasets
df_raw, merge_logs = merge_hr_datasets(raw_datasets)

if df_raw is None or df_raw.empty:
    st.error("Failed to load or merge HR datasets. Please verify dataset schemas.")
    st.stop()

# Feature engineering (burnout index, compa-ratio, tenure cohorts, flight risk factors)
df_workforce, engineered_features = engineer_features(df_raw)

# Sidebar Interactive Filters
all_departments = sorted(df_workforce["department"].dropna().unique().tolist()) if "department" in df_workforce.columns else []
all_job_levels = sorted(df_workforce["job_level"].dropna().unique().tolist()) if "job_level" in df_workforce.columns else []

with st.sidebar:
    st.markdown("### 🔎 Workforce Filters")
    selected_depts = st.multiselect("Departments", options=all_departments, default=all_departments)
    selected_levels = st.multiselect("Job Levels", options=all_job_levels, default=all_job_levels)
    
    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()

# Apply Filters
df_active_filtered = df_workforce.copy()
if selected_depts and "department" in df_active_filtered.columns:
    df_active_filtered = df_active_filtered[df_active_filtered["department"].isin(selected_depts)]
if selected_levels and "job_level" in df_active_filtered.columns:
    df_active_filtered = df_active_filtered[df_active_filtered["job_level"].isin(selected_levels)]

if df_active_filtered.empty:
    st.warning("No workforce records match the selected filter criteria. Please broaden your selection.")
    st.stop()


# -----------------------------------------------------------------------------
# CORE ANALYTICS CALCULATIONS
# -----------------------------------------------------------------------------
kpis = calculate_executive_kpis(df_active_filtered)
dept_summary = get_department_breakdown(df_active_filtered)
monthly_trends = get_monthly_attrition_trends(df_active_filtered)
tenure_cohorts = get_tenure_cohort_analysis(df_active_filtered)
stat_tests = perform_group_comparison_tests(df_active_filtered)
cat_tests = perform_categorical_chi2_tests(df_active_filtered)
turnover_costs = calculate_turnover_financials(df_active_filtered)


# -----------------------------------------------------------------------------
# TOP HERO BANNER
# -----------------------------------------------------------------------------
render_hero(
    title="PeoplePulse AI — Strategic Attrition & Retention Intelligence",
    subtitle=f"Enterprise workforce analytics for {kpis.get('total_employees', 0):,} evaluated personnel | "
             f"{kpis.get('overall_attrition_rate', 0)}% Annualized Attrition | "
             f"${turnover_costs.get('total_turnover_cost', 0):,.0f} Estimated Turnover Drag",
    badge_text="LIVE ANALYTICS ENGINE"
)


# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_overview, tab_stats, tab_ml, tab_costs, tab_playbook, tab_insights, tab_reports = st.tabs([
    "📊 Executive Overview",
    "🔍 Statistical Root-Causes",
    "🤖 Predictive ML & Flight Risk",
    "💰 Turnover Financials & ROI",
    "🛡️ Retention Playbook & Actions",
    "🧠 LLM Strategic Briefing",
    "📄 Executive Dossier & Export"
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tab_overview:
    # KPI Cards Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    render_kpi(
        kpi_col1,
        "Total Headcount",
        f"{kpis.get('total_employees', 0):,}",
        f"Active: {kpis.get('active_employees', 0):,}",
        trend=f"Exits: {kpis.get('exited_employees', 0)}"
    )
    
    render_kpi(
        kpi_col2,
        "Overall Attrition Rate",
        f"{kpis.get('overall_attrition_rate', 0.0)}%",
        f"Voluntary: {kpis.get('voluntary_attrition_rate', 0.0)}%",
        trend=f"Avg HC: {kpis.get('avg_headcount', 0):,.0f}"
    )
    
    render_kpi(
        kpi_col3,
        "Avg Tenure & Absence",
        f"{kpis.get('avg_tenure_years', 0.0)} yrs",
        f"Absenteeism: {kpis.get('avg_absenteeism_days', 0.0)} d/yr",
        trend=f"OT: {kpis.get('overtime_percentage', 0.0)}%"
    )
    
    render_kpi(
        kpi_col4,
        "Avg Engagement Score",
        f"{kpis.get('avg_engagement_score', 0.0)}/100",
        f"Perf Rating: {kpis.get('avg_performance_rating', 0.0)}/5.0",
        trend="Index"
    )
    
    render_kpi(
        kpi_col5,
        "Estimated Turnover Cost",
        f"${turnover_costs.get('total_turnover_cost', 0):,.0f}",
        f"Avg/Exit: ${turnover_costs.get('avg_cost_per_exit', 0):,.0f}",
        trend=f"{turnover_costs.get('num_exits', 0)} exits"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row 1
    col_chart1, col_chart2 = st.columns([1.2, 1])
    
    with col_chart1:
        st.markdown("#### 🏢 Departmental Attrition vs Benchmark Corridor")
        if not dept_summary.empty:
            fig_dept = px.bar(
                dept_summary,
                x="Attrition_Rate_%",
                y="department",
                orientation="h",
                text="Attrition_Rate_%",
                color="Attrition_Rate_%",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                labels={"Attrition_Rate_%": "Attrition Rate (%)", "department": "Department"}
            )
            fig_dept.add_vline(
                x=kpis.get("overall_attrition_rate", 15.0),
                line_dash="dash",
                line_color="#38BDF8",
                annotation_text=f"Org Avg: {kpis.get('overall_attrition_rate', 0)}%",
                annotation_position="top right"
            )
            fig_dept.update_layout(get_plotly_layout())
            fig_dept.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_dept, use_container_width=True)
        else:
            st.info("Insufficient department data for plotting.")
            
    with col_chart2:
        st.markdown("#### 📈 Historical Departure Velocity")
        if not monthly_trends.empty and "Total_Exits" in monthly_trends.columns:
            fig_trend = px.area(
                monthly_trends,
                x="YearMonth",
                y="Total_Exits",
                markers=True,
                labels={"YearMonth": "Month", "Total_Exits": "Exits Recorded"},
                color_discrete_sequence=["#EF4444"]
            )
            fig_trend.update_layout(get_plotly_layout())
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            # Fallback distribution
            fig_pie = px.pie(
                df_active_filtered,
                names="department",
                title="Workforce Distribution by Department",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(get_plotly_layout())
            st.plotly_chart(fig_pie, use_container_width=True)
            
    # Visualizations Row 2
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.markdown("#### ⏳ Tenure Cohort Attrition (Stay Curve)")
        if not tenure_cohorts.empty:
            fig_cohort = px.bar(
                tenure_cohorts,
                x="tenure_cohort",
                y="Attrition_Rate_%",
                text="Attrition_Rate_%",
                color="Attrition_Rate_%",
                color_continuous_scale="Blues",
                labels={"tenure_cohort": "Tenure Cohort", "Attrition_Rate_%": "Cohort Attrition %"}
            )
            fig_cohort.update_layout(get_plotly_layout())
            fig_cohort.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_cohort, use_container_width=True)
        else:
            st.info("Tenure cohort analysis requires tenure features.")
            
    with col_c4:
        st.markdown("#### ⚖️ Role Compa-Ratio vs Overtime Dispersion")
        if "compa_ratio" in df_active_filtered.columns and "overtime_hours" in df_active_filtered.columns:
            fig_scatter = px.scatter(
                df_active_filtered,
                x="compa_ratio",
                y="overtime_hours",
                color="attrition_label",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={"compa_ratio": "Compa-Ratio Benchmark", "overtime_hours": "Weekly Overtime Hours", "attrition_label": "Departed"},
                hover_data=["employee_id", "department", "job_role"] if "job_role" in df_active_filtered.columns else ["employee_id"],
                opacity=0.7
            )
            fig_scatter.add_vline(x=0.90, line_dash="dot", line_color="#F59E0B", annotation_text="Underpaid Threshold (<0.90)")
            fig_scatter.add_hline(y=8.0, line_dash="dot", line_color="#EF4444", annotation_text="High Overtime (≥8h/wk)")
            fig_scatter.update_layout(get_plotly_layout())
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    # Department Breakdown Table
    st.markdown("#### 📋 Comprehensive Departmental Summary Table")
    if not dept_summary.empty:
        st.dataframe(
            dept_summary.style.format({
                "Total_Employees": "{:,}",
                "Exits": "{:,}",
                "Active": "{:,}",
                "Avg_Headcount": "{:.1f}",
                "Attrition_Rate_%": "{:.1f}%",
                "Avg_Engagement": "{:.1f}",
                "Avg_Overtime": "{:.1f}h",
                "Avg_Compa": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 2: STATISTICAL ROOT-CAUSE ANALYSIS
# =============================================================================
with tab_stats:
    st.markdown("### 🔬 Rigorous Hypothesis Testing & Statistical Driver Identification")
    render_callout(
        "Empirical differences between Exited and Retained personnel are evaluated using independent two-sample "
        "t-tests, non-parametric Mann-Whitney U tests, Cohen's d effect sizes, and multivariable logistic regression. "
        "Associations indicate observable systemic patterns and should be contextualized before making policy interventions.",
        kind="info"
    )
    
    col_st1, col_st2 = st.columns([1.5, 1])
    
    with col_st1:
        st.markdown("#### 📊 Continuous Factor Differences (Exited vs Retained)")
        if stat_tests:
            df_tests = pd.DataFrame(stat_tests)
            st.dataframe(
                df_tests[[
                    "metric_name", "active_mean", "exited_mean", "mean_difference",
                    "direction", "p_value", "cohens_d", "effect_magnitude", "is_significant"
                ]].style.format({
                    "active_mean": "{:.2f}",
                    "exited_mean": "{:.2f}",
                    "mean_difference": "{:+.2f}",
                    "p_value": "{:.4e}",
                    "cohens_d": "{:.2f}"
                }).map(
                    lambda v: "color: #10B981; font-weight: bold;" if v is True else "color: #94A3B8;",
                    subset=["is_significant"]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Insufficient data to perform two-sample statistical tests.")
            
    with col_st2:
        st.markdown("#### 🏷️ Categorical Associations (Chi-Square & Cramér's V)")
        if cat_tests:
            df_cat = pd.DataFrame(cat_tests)
            st.dataframe(
                df_cat.style.format({
                    "chi2_stat": "{:.2f}",
                    "p_value": "{:.4e}",
                    "cramers_v": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Categorical association tests require discrete categorical columns.")
            
    st.markdown("---")
    
    # Multivariable Logistic Regression Table
    st.markdown("#### 📐 Multivariable Logistic Regression (Adjusted Odds Ratios with 95% CI)")
    reg_summary, reg_metrics = fit_multivariable_logistic_regression(df_active_filtered)
    
    if reg_summary is not None and isinstance(reg_summary, pd.DataFrame):
        col_reg1, col_reg2 = st.columns([2, 1])
        with col_reg1:
            st.dataframe(
                reg_summary[[
                    "label", "coefficient", "odds_ratio", "ci_95_lower", "ci_95_upper", "p_value", "is_significant"
                ]].style.format({
                    "coefficient": "{:+.3f}",
                    "odds_ratio": "{:.3f}",
                    "ci_95_lower": "{:.3f}",
                    "ci_95_upper": "{:.3f}",
                    "p_value": "{:.4e}"
                }).map(
                    lambda v: "background-color: rgba(239, 68, 68, 0.15); color: #F87171; font-weight: bold;" if v > 1.25
                    else ("background-color: rgba(16, 185, 129, 0.15); color: #34D399; font-weight: bold;" if v < 0.80 else ""),
                    subset=["odds_ratio"]
                ),
                use_container_width=True,
                hide_index=True
            )
        with col_reg2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Model Goodness-of-Fit</div>
                <div class="kpi-value">Pseudo R²: {reg_metrics.get('prsquared', 0.0)}</div>
                <div class="kpi-subtext">Observations Analyzed: <strong>{reg_metrics.get('nobs', 0)}</strong></div>
                <div class="kpi-subtext">LLR p-value: <strong>{reg_metrics.get('llr_pvalue', 0.0):.4e}</strong></div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 10px;">
                    Odds Ratio > 1.0 indicates elevated risk of departure. Odds Ratio < 1.0 indicates protective factor.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"Regression notice: {reg_metrics}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Distribution Explorer
    st.markdown("#### 🎻 Interactive Distribution Explorer")
    numeric_choices = [c for c in [
        "overtime_hours", "compa_ratio", "manager_satisfaction",
        "work_life_balance_score", "engagement_score", "absenteeism_days",
        "weekly_working_hours", "training_hours", "tenure_years"
    ] if c in df_active_filtered.columns]
    
    selected_num = st.selectbox("Select Metric for Group Distribution Comparison", numeric_choices)
    if selected_num:
        fig_dist = px.violin(
            df_active_filtered,
            y=selected_num,
            x="attrition_label",
            color="attrition_label",
            box=True,
            points="outliers",
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            labels={"attrition_label": "Status (0 = Retained, 1 = Exited)", selected_num: selected_num.replace("_", " ").title()}
        )
        fig_dist.update_layout(get_plotly_layout())
        st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# TAB 3: PREDICTIVE ML & FLIGHT RISK ENGINE
# =============================================================================
with tab_ml:
    st.markdown("### 🤖 Predictive Modeling & Flight Risk Attribution Engine")
    
    # Train / Retrieve Models
    if st.session_state.ml_context is None:
        with st.spinner("Training Logistic Regression, Random Forest, and Gradient Boosting pipelines..."):
            ml_ctx, ml_err = train_and_evaluate_models(df_workforce, test_size=0.20, random_state=42)
            if ml_err:
                st.error(f"Modeling error: {ml_err}")
            else:
                st.session_state.ml_context = ml_ctx
                
    ml_context = st.session_state.ml_context
    
    if ml_context and "results" in ml_context:
        # Comparison Metrics Table
        model_results = ml_context["results"]
        metrics_table = []
        for m_name, m_data in model_results.items():
            row = {"Model Pipeline": m_name}
            row.update(m_data["metrics"])
            metrics_table.append(row)
            
        st.markdown("#### 🏆 Supervised Model Benchmark on Out-of-Sample Test Set (20% Holdout)")
        st.dataframe(
            pd.DataFrame(metrics_table).style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1_Score": "{:.3f}",
                "ROC_AUC": "{:.3f}",
                "PR_AUC": "{:.3f}",
                "Brier_Score": "{:.3f}"
            }).highlight_max(subset=["ROC_AUC", "PR_AUC", "F1_Score"], color="rgba(59, 130, 246, 0.3)"),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Selected Model Deep Dive
        selected_model_name = st.selectbox(
            "Select Model for Detailed Inspection & Explanations",
            list(model_results.keys()),
            index=1 # default to Random Forest
        )
        
        active_pipeline = model_results[selected_model_name]["pipeline"]
        y_test = model_results[selected_model_name]["y_test"]
        y_prob = model_results[selected_model_name]["y_prob"]
        
        col_m1, col_m2 = st.columns([1, 1.2])
        
        with col_m1:
            st.markdown("#### 🎯 Interactive Decision Threshold Tuning")
            st.caption("Adjust the classification threshold to prioritize Recall (capturing all flight risks) or Precision (minimizing false alarms).")
            
            thresh_slider = st.slider(
                "Classification Probability Threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(st.session_state.decision_threshold),
                step=0.05
            )
            st.session_state.decision_threshold = thresh_slider
            
            thresh_eval = evaluate_custom_threshold(y_test, y_prob, threshold=thresh_slider)
            
            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.metric("Precision", f"{thresh_eval['precision']:.3f}")
            t_col2.metric("Recall", f"{thresh_eval['recall']:.3f}")
            t_col3.metric("F1-Score", f"{thresh_eval['f1']:.3f}")
            
            # Confusion Matrix Visualization
            cm = np.array(thresh_eval["cm"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Retained (0)", "Exited (1)"],
                y=["Retained (0)", "Exited (1)"],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(get_plotly_layout())
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("#### 🔑 Model-Agnostic Feature Importance")
            perm_imp, perm_msg = compute_permutation_importance(
                active_pipeline,
                ml_context["X_test"],
                ml_context["y_test"],
                ml_context["available_num"],
                ml_context["available_cat"]
            )
            
            if not perm_imp.empty:
                top_features = perm_imp.head(10)
                fig_perm = px.bar(
                    top_features,
                    x="Importance_Mean",
                    y="Feature",
                    orientation="h",
                    color="Importance_Mean",
                    color_continuous_scale="Blues",
                    labels={"Importance_Mean": "Drop in ROC-AUC when Shuffled", "Feature": "Feature"}
                )
                fig_perm.update_layout(get_plotly_layout())
                st.plotly_chart(fig_perm, use_container_width=True)
            else:
                st.info(perm_msg)
                
        st.markdown("---")
        
        # Workforce Flight Risk Scoring
        st.markdown("#### 👥 Active Workforce Flight Risk Explorer")
        st.caption("Active personnel scored using the calibrated predictive model. Categorized into Elevated (≥60%), Moderate (35-59%), and Low (<35%) risk tiers.")
        
        df_scored = score_entire_workforce(active_pipeline, df_active_filtered, threshold=thresh_slider)
        
        # Risk Tier Filter
        tier_filter = st.radio(
            "Filter Risk Tier",
            ["All Tiers", "Elevated Risk", "Moderate Risk", "Low Risk"],
            horizontal=True
        )
        
        df_display_risk = df_scored[df_scored["attrition_label"] == 0].copy()
        if tier_filter != "All Tiers":
            df_display_risk = df_display_risk[df_display_risk["risk_tier"] == tier_filter]
            
        display_cols = [c for c in [
            "employee_id", "department", "job_role", "job_level",
            "predicted_attrition_risk", "risk_tier", "overtime_hours",
            "compa_ratio", "manager_satisfaction", "engagement_score"
        ] if c in df_display_risk.columns]
        
        st.dataframe(
            df_display_risk[display_cols].sort_values(by="predicted_attrition_risk", ascending=False).style.format({
                "predicted_attrition_risk": "{:.3f}",
                "overtime_hours": "{:.1f}h",
                "compa_ratio": "{:.2f}",
                "manager_satisfaction": "{:.1f}",
                "engagement_score": "{:.1f}"
            }).map(
                lambda v: "background-color: rgba(239, 68, 68, 0.2); color: #F87171; font-weight: bold;" if v == "Elevated Risk"
                else ("background-color: rgba(245, 158, 11, 0.2); color: #FBBF24;" if v == "Moderate Risk" else "color: #34D399;"),
                subset=["risk_tier"] if "risk_tier" in display_cols else []
            ),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 4: TURNOVER FINANCIALS & WHAT-IF ROI SIMULATOR
# =============================================================================
with tab_costs:
    st.markdown("### 💰 Turnover Financial Drag & Interactive What-If ROI Simulator")
    
    col_fin1, col_fin2 = st.columns([1, 1.2])
    
    with col_fin1:
        st.markdown("#### 💸 Comprehensive Turnover Cost Breakdown")
        st.caption("Includes recruitment sourcing, onboarding, training, vacancy coverage, and ramp-up drag.")
        
        cost_bd = turnover_costs.get("cost_breakdown", {})
        if cost_bd:
            df_bd = pd.DataFrame(list(cost_bd.items()), columns=["Cost Category", "Expense ($)"])
            fig_bd = px.pie(
                df_bd,
                names="Cost Category",
                values="Expense ($)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_bd.update_layout(get_plotly_layout())
            st.plotly_chart(fig_bd, use_container_width=True)
            
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Financial Drag</div>
                <div class="kpi-value">${turnover_costs.get('total_turnover_cost', 0):,.0f}</div>
                <div class="kpi-subtext">Average Replacement Cost / Departure: <strong>${turnover_costs.get('avg_cost_per_exit', 0):,.0f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_fin2:
        st.markdown("#### 🏢 Departmental Turnover Expense Impact")
        dept_costs = turnover_costs.get("dept_costs", pd.DataFrame())
        if not dept_costs.empty:
            fig_dc = px.bar(
                dept_costs,
                x="total_cost",
                y="department",
                orientation="h",
                text="total_cost",
                color="total_cost",
                color_continuous_scale="Reds",
                labels={"total_cost": "Total Turnover Drag ($)", "department": "Department"}
            )
            fig_dc.update_layout(get_plotly_layout())
            fig_dc.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            st.plotly_chart(fig_dc, use_container_width=True)
            
    st.markdown("---")
    
    # What-If Scenario Simulator
    st.markdown("#### 🎛️ Interactive Retention Scenario Simulator")
    render_callout(
        "Simulate hypothetical organizational interventions across Overtime, Compensation, Culture, and Training. "
        "Projected savings are computed using empirical regression elasticities and benchmarked replacement costs. "
        "Results represent scenario projections, not causal guarantees.",
        kind="warning"
    )
    
    sim_col1, sim_col2 = st.columns([1, 1.2])
    
    with sim_col1:
        st.markdown("##### ⚙️ Intervention Parameters")
        sim_ot = st.slider("Overtime Reduction in High-OT Cohort (%)", 0.0, 50.0, 30.0, 5.0)
        sim_comp = st.slider("Underpaid Compa-Ratio Adjustment (%)", 0.0, 30.0, 15.0, 2.5)
        sim_eng = st.slider("Cultural Engagement Boost (points)", 0.0, 15.0, 5.0, 1.0)
        sim_train = st.slider("Training & Upskilling Expansion (%)", 0.0, 50.0, 25.0, 5.0)
        sim_budget = st.number_input("Total Estimated Program Investment ($)", min_value=10000.0, max_value=500000.0, value=120000.0, step=10000.0)
        
    with sim_col2:
        st.markdown("##### 📈 Projected Return on Investment (ROI)")
        sim_results = simulate_retention_scenario(
            df_active_filtered,
            cost_per_exit=turnover_costs.get("avg_cost_per_exit", 25000.0),
            ot_reduction_pct=sim_ot,
            comp_adjustment_pct=sim_comp,
            engagement_boost_pts=sim_eng,
            training_expansion_pct=sim_train,
            est_intervention_budget=sim_budget
        )
        
        s_c1, s_c2, s_c3 = st.columns(3)
        s_c1.metric("Projected Exits Avoided", f"{sim_results.get('projected_exits_saved', 0):,} exits")
        s_c2.metric("Turnover Cost Avoided", f"${sim_results.get('avoided_turnover_cost', 0):,.0f}")
        s_c3.metric("Net Financial ROI", f"{sim_results.get('roi_pct', 0.0):.1f}%", delta=f"${sim_results.get('net_financial_impact', 0):,.0f}")
        
        # New projected attrition rate comparison
        st.markdown(f"""
        <div class="kpi-card" style="margin-top: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="kpi-title">Baseline vs Projected Attrition Rate</div>
                    <div style="font-size: 20px; font-weight: 700; color: #F8FAFC;">
                        {sim_results.get('baseline_attrition_rate', 0)}% ➔ <span style="color: #34D399;">{sim_results.get('projected_attrition_rate', 0)}%</span>
                    </div>
                </div>
                <div class="badge-low" style="font-size: 14px; padding: 6px 14px;">
                    -{sim_results.get('rate_reduction_points', 0)} Percentage Points
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Savings breakdown by initiative
        saved_bd = sim_results.get("exits_saved_breakdown", {})
        if saved_bd:
            df_sbd = pd.DataFrame(list(saved_bd.items()), columns=["Initiative", "Projected Exits Saved"])
            fig_sbd = px.bar(
                df_sbd,
                x="Projected Exits Saved",
                y="Initiative",
                orientation="h",
                text="Projected Exits Saved",
                color="Projected Exits Saved",
                color_continuous_scale="Greens"
            )
            fig_sbd.update_layout(get_plotly_layout())
            st.plotly_chart(fig_sbd, use_container_width=True)


# =============================================================================
# TAB 5: RETENTION PLAYBOOK & ACTION TRACKER
# =============================================================================
with tab_playbook:
    st.markdown("### 🛡️ Evidence-Based Retention Playbook & Action Tracker")
    
    # 1. Automated Recommendations
    st.markdown("#### 💡 Empirical Retention Interventions (Ranked by Impact)")
    recs = generate_evidence_based_recommendations(df_active_filtered, stat_tests)
    
    if recs:
        for rec in recs:
            with st.container():
                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom: 12px; border-left: 4px solid {PRIMARY_COLOR};">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 11px; font-weight: 700; color: #60A5FA; text-transform: uppercase;">{rec['pillar']}</span>
                            <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{rec['issue']}</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="badge-elevated">Impact Score: {rec['impact_score']}/100</span>
                            <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Feasibility: <strong>{rec['feasibility']}</strong></div>
                        </div>
                    </div>
                    <div style="font-size: 13px; color: #CBD5E1; margin-top: 8px;">
                        <strong>Empirical Evidence:</strong> {rec['evidence']}
                    </div>
                    <div style="font-size: 13px; color: #E2E8F0; margin-top: 6px;">
                        <strong>Proposed Action:</strong> {rec['proposed_intervention']}
                    </div>
                    <div style="display: flex; gap: 20px; font-size: 11px; color: #94A3B8; margin-top: 8px;">
                        <div>Owner: <strong>{rec['responsible_function']}</strong></div>
                        <div>Timeline: <strong>{rec['timeline']}</strong></div>
                        <div>Success Target: <strong>{rec['success_metric']}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Button to 1-click add to tracker
                if st.button(f"➕ Add to SQLite Action Tracker: {rec['id']}", key=f"btn_add_{rec['id']}"):
                    add_retention_action(
                        title=rec['issue'],
                        pillar=rec['pillar'],
                        department=rec['affected_population'],
                        owner=rec['responsible_function'],
                        issue=rec['evidence'],
                        action=rec['proposed_intervention'],
                        kpi_name=rec['success_metric'],
                        kpi_base="Baseline",
                        kpi_target=rec['timeline'],
                        review_date="2026-06-30",
                        status="Planned"
                    )
                    st.success(f"Initiative '{rec['id']}' added to Action Tracker!")
                    st.rerun()
    else:
        st.info("No urgent intervention triggers identified in active population.")
        
    st.markdown("---")
    
    # 2. SQLite Action Tracker Table
    st.markdown("#### 📋 Retention Action Tracker (SQLite Database)")
    df_actions = get_all_actions()
    
    if not df_actions.empty:
        st.dataframe(
            df_actions[[
                "id", "title", "pillar", "department", "owner",
                "status", "baseline_kpi_name", "target_kpi_val", "followup_kpi_val", "review_date"
            ]].style.map(
                lambda v: "background-color: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: bold;" if v == "Completed"
                else ("background-color: rgba(59, 130, 246, 0.2); color: #60A5FA;" if v == "In Progress" else ""),
                subset=["status"]
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No retention actions currently tracked. Add one below!")
        
    # Interactive CRUD Expander
    with st.expander("📝 Manage Retention Initiatives (Add / Update / Delete)", expanded=False):
        crud_tab1, crud_tab2, crud_tab3 = st.tabs(["Add New Action", "Update Status", "Delete Action"])
        
        with crud_tab1:
            with st.form("form_add_action"):
                f_title = st.text_input("Initiative Title", "Engineering Burnout Reduction Sprints")
                f_pillar = st.selectbox("Strategic Pillar", ["Workload & Wellbeing", "Compensation & Career Growth", "Engagement & Leadership", "Learning & Development"])
                f_dept = st.text_input("Target Department / Cohort", "Engineering")
                f_owner = st.text_input("Designated Owner", "VP of Engineering & HRBP")
                f_issue = st.text_area("Identified Issue", "Sustained weekly overtime >10 hrs in backend team.")
                f_action = st.text_area("Proposed Action", "Cap overtime at 8 hrs/wk and introduce recovery comp days.")
                f_kpi = st.text_input("Success KPI Name", "Average Weekly Overtime Hours")
                f_base = st.text_input("Baseline KPI Value", "10.4 hrs/wk")
                f_target = st.text_input("Target KPI Value", "< 5.0 hrs/wk")
                f_date = st.date_input("Target Review Date", datetime(2026, 6, 30))
                
                if st.form_submit_button("Save New Retention Action"):
                    add_retention_action(
                        f_title, f_pillar, f_dept, f_owner, f_issue, f_action,
                        f_kpi, f_base, f_target, str(f_date), status="Planned"
                    )
                    st.success("New retention action saved to database!")
                    st.rerun()
                    
        with crud_tab2:
            if not df_actions.empty:
                with st.form("form_update_action"):
                    u_id = st.selectbox("Select Action to Update", df_actions["id"].tolist(), format_func=lambda x: f"ID {x}: {df_actions[df_actions['id']==x]['title'].values[0]}")
                    u_status = st.selectbox("New Status", ["Proposed", "Approved", "In Progress", "Under Review", "Completed", "On Hold"])
                    u_followup = st.text_input("Follow-up KPI Value", "")
                    u_notes = st.text_area("Evaluation Notes / Progress", "")
                    
                    if st.form_submit_button("Update Action Status"):
                        update_action_status(u_id, u_status, followup_kpi=u_followup, notes=u_notes)
                        st.success(f"Action #{u_id} updated successfully!")
                        st.rerun()
            else:
                st.caption("No action items available to update.")
                
        with crud_tab3:
            if not df_actions.empty:
                with st.form("form_delete_action"):
                    d_id = st.selectbox("Select Action to Delete", df_actions["id"].tolist(), format_func=lambda x: f"ID {x}: {df_actions[df_actions['id']==x]['title'].values[0]}")
                    if st.form_submit_button("Confirm Permanent Deletion", type="primary"):
                        delete_retention_action(d_id)
                        st.success(f"Action #{d_id} removed.")
                        st.rerun()


# =============================================================================
# TAB 6: LLM STRATEGIC BRIEFING
# =============================================================================
with tab_insights:
    st.markdown("### 🧠 LLM Executive Intelligence & Strategic Evidence Synthesis")
    st.caption("Synthesizes empirical findings, odds ratios, and retention priorities into an executive-ready strategic briefing.")
    
    if st.button("🚀 Generate Executive Intelligence Briefing", type="primary"):
        with st.spinner("Synthesizing empirical workforce signals into executive intelligence..."):
            top_factors = []
            if ml_context and "results" in ml_context:
                rf_res = ml_context["results"].get("Random Forest (Nonlinear)")
                if rf_res:
                    p_imp, _ = compute_permutation_importance(
                        rf_res["pipeline"], ml_context["X_test"], ml_context["y_test"],
                        ml_context["available_num"], ml_context["available_cat"]
                    )
                    if not p_imp.empty:
                        top_factors = p_imp["Feature"].head(5).tolist()
                        
            insights, engine_source = query_llm_for_hr_insights(
                kpis, stat_tests, reg_summary, top_factors,
                api_key=st.session_state.openai_api_key
            )
            
            st.session_state.cached_insights = insights
            st.session_state.insights_source = engine_source
            
    if "cached_insights" in st.session_state:
        st.markdown(f"**Intelligence Engine:** `{st.session_state.get('insights_source', 'Rule-Based Engine')}`")
        st.markdown("<br>", unsafe_allow_html=True)
        
        for ins in st.session_state.cached_insights:
            with st.container():
                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom: 16px; border-left: 4px solid {SECONDARY_COLOR};">
                    <div style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">{ins.get('title', 'Workforce Insight')}</div>
                    <div style="font-size: 14px; color: #E2E8F0; margin-bottom: 10px;"><strong>Executive Finding:</strong> {ins.get('executive_finding', '')}</div>
                    <div style="font-size: 13px; color: #94A3B8; margin-bottom: 6px;"><strong>Supporting Evidence:</strong> {ins.get('supporting_evidence', '')}</div>
                    <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 6px;"><strong>Organizational Dynamics:</strong> {ins.get('organizational_explanation', '')}</div>
                    <div style="font-size: 13px; color: #60A5FA; margin-bottom: 6px;"><strong>Affected Segment:</strong> {ins.get('affected_segment', '')}</div>
                    <div style="font-size: 12px; color: #FBBF24; margin-bottom: 6px;"><strong>Confidence & Caveats:</strong> {ins.get('confidence_and_limitations', '')}</div>
                    <div style="background: rgba(59, 130, 246, 0.1); border-radius: 6px; padding: 10px; margin-top: 10px;">
                        <div style="font-size: 13px; color: #F8FAFC; font-weight: 600;">Recommended Immediate Action:</div>
                        <div style="font-size: 12.5px; color: #E2E8F0;">{ins.get('suggested_intervention', '')}</div>
                        <div style="font-size: 11px; color: #38BDF8; margin-top: 4px;">KPI to Monitor: <strong>{ins.get('kpi_to_monitor', '')}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Executive Intelligence Briefing' above to run the synthesis engine.")


# =============================================================================
# TAB 7: EXECUTIVE DOSSIER & EXPORT
# =============================================================================
with tab_reports:
    st.markdown("### 📄 Executive Decision-Support Dossier & Data Export")
    st.caption("Generate formal board-level management PDFs and comprehensive multi-sheet Excel reports.")
    
    col_rep1, col_rep2, col_rep3 = st.columns(3)
    
    with col_rep1:
        st.markdown("#### 📕 Executive PDF Dossier")
        st.caption("Includes formal ReportLab typography, workforce KPIs, statistical tests, model metrics, and responsible AI disclosures.")
        
        pdf_bytes = generate_pdf_management_report(
            kpi_data=kpis,
            statistical_tests=stat_tests,
            model_metrics=ml_context["results"] if ml_context else {},
            recommendations=recs,
            turnover_costs=turnover_costs
        )
        
        st.download_button(
            "📥 Download Executive PDF Report",
            data=pdf_bytes,
            file_name=f"PeoplePulse_AI_Executive_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        
    with col_rep2:
        st.markdown("#### 📗 Multi-Sheet Excel Workbook")
        st.caption("5 structured sheets: Executive_KPIs, Department_Breakdown, Model_Validation, Retention_Interventions, Turnover_Financials.")
        
        excel_bytes = export_multi_sheet_excel(
            kpis=kpis,
            dept_breakdown=dept_summary,
            model_metrics=ml_context["results"] if ml_context else {},
            recommendations=recs,
            turnover_costs=turnover_costs
        )
        
        st.download_button(
            "📥 Download Multi-Sheet Excel Workbook",
            data=excel_bytes,
            file_name=f"PeoplePulse_AI_Workforce_Data_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    with col_rep3:
        st.markdown("#### 📊 Filtered Workforce CSV")
        st.caption("Export the currently filtered active workforce records with all engineered features.")
        
        csv_buffer = io.StringIO()
        df_active_filtered.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "📥 Download Filtered Workforce CSV",
            data=csv_buffer.getvalue(),
            file_name=f"PeoplePulse_Filtered_Workforce_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.markdown("---")
    render_callout(
        "Responsible AI Disclosure: PeoplePulse AI is designed exclusively for aggregate organizational intelligence, "
        "turnover cost reduction, and systemic policy optimization. It is not intended for adverse automated individual employment "
        "actions. All predictive scores must be contextualized with qualitative management review.",
        kind="info"
    )

