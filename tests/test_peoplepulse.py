"""
Unit Tests for PeoplePulse AI Platform
Tests data loading, schema validation, feature engineering, attrition rate math,
machine learning models, turnover cost calculations, and SQLite action tracker.
"""

import os
import pytest
import pandas as pd
import numpy as np

from modules.data_loader import load_default_datasets, merge_hr_datasets, DATASET_SCHEMAS
from modules.data_validation import validate_schema, inspect_dataset_quality
from modules.feature_engineering import engineer_features
from modules.attrition_analysis import calculate_executive_kpis, get_department_breakdown
from modules.statistical_analysis import perform_group_comparison_tests, fit_multivariable_logistic_regression
from modules.prediction_models import train_and_evaluate_models, evaluate_custom_threshold
from modules.cost_analysis import calculate_turnover_financials, calculate_retention_roi
from modules.scenario_simulator import simulate_retention_scenario
from modules.action_tracker import init_db, add_retention_action, get_all_actions, update_action_status, delete_retention_action

def test_data_loader_and_schemas():
    """Verify that sample datasets exist and load properly."""
    datasets = load_default_datasets("data")
    assert "employees" in datasets
    assert datasets["employees"] is not None
    assert len(datasets["employees"]) >= 1000
    
    # Test schema validation
    is_valid, missing_req, missing_opt, mapping = validate_schema(
        datasets["employees"], "employees", DATASET_SCHEMAS["employees"]
    )
    assert is_valid is True
    assert len(missing_req) == 0

def test_data_merge_and_feature_engineering():
    """Verify merging and feature engineering pipeline without leakage."""
    datasets = load_default_datasets("data")
    df_merged, logs = merge_hr_datasets(datasets)
    assert df_merged is not None
    assert len(df_merged) >= 1000
    
    df_engineered, features = engineer_features(df_merged)
    assert "attrition_label" in df_engineered.columns
    assert "compa_ratio" in df_engineered.columns
    assert "tenure_years" in df_engineered.columns
    assert "overtime_hours" in df_engineered.columns
    
    # Check that compa_ratio is reasonable positive float
    assert (df_engineered["compa_ratio"] > 0).all()
    # Check target labels are 0 or 1
    assert set(df_engineered["attrition_label"].unique()).issubset({0, 1})

def test_attrition_kpi_calculations():
    """Verify standard HR attrition formula: Exits / (Active + Exits / 2) * 100."""
    datasets = load_default_datasets("data")
    df_merged, _ = merge_hr_datasets(datasets)
    df_eng, _ = engineer_features(df_merged)
    
    kpis = calculate_executive_kpis(df_eng)
    assert kpis["total_employees"] == len(df_eng)
    assert kpis["active_employees"] + kpis["exited_employees"] == kpis["total_employees"]
    
    # Check rate calculation
    expected_rate = round(
        kpis["exited_employees"] / (kpis["active_employees"] + kpis["exited_employees"] / 2.0) * 100.0, 1
    )
    assert kpis["overall_attrition_rate"] == expected_rate

def test_statistical_analysis():
    """Verify t-tests and effect size calculations."""
    datasets = load_default_datasets("data")
    df_merged, _ = merge_hr_datasets(datasets)
    df_eng, _ = engineer_features(df_merged)
    
    tests = perform_group_comparison_tests(df_eng)
    assert len(tests) > 0
    top_test = tests[0]
    assert "metric_name" in top_test
    assert "p_value" in top_test
    assert "cohens_d" in top_test

def test_ml_prediction_models():
    """Verify Logistic Regression, Random Forest, and Gradient Boosting pipelines."""
    datasets = load_default_datasets("data")
    df_merged, _ = merge_hr_datasets(datasets)
    df_eng, _ = engineer_features(df_merged)
    
    context, err = train_and_evaluate_models(df_eng, test_size=0.2, random_state=42)
    assert err is None
    assert "Logistic Regression (Baseline)" in context["results"]
    assert "Random Forest (Nonlinear)" in context["results"]
    assert "Gradient Boosting (Ensemble)" in context["results"]
    
    rf_metrics = context["results"]["Random Forest (Nonlinear)"]["metrics"]
    assert rf_metrics["ROC_AUC"] > 0.65
    assert "Precision" in rf_metrics
    assert "Recall" in rf_metrics

def test_turnover_cost_and_scenario_simulation():
    """Verify financial cost modeling and scenario savings math."""
    datasets = load_default_datasets("data")
    df_merged, _ = merge_hr_datasets(datasets)
    df_eng, _ = engineer_features(df_merged)
    
    financials = calculate_turnover_financials(df_eng)
    assert financials["num_exits"] > 0
    assert financials["total_turnover_cost"] > 0
    assert financials["avg_cost_per_exit"] > 10000
    
    sim = simulate_retention_scenario(
        df_eng,
        cost_per_exit=financials["avg_cost_per_exit"],
        ot_reduction_pct=30.0,
        comp_adjustment_pct=20.0,
        est_intervention_budget=100000.0
    )
    assert sim["projected_exits_saved"] >= 0
    assert sim["projected_turnover_cost"] <= sim["baseline_turnover_cost"]
    assert "net_financial_impact" in sim

def test_sqlite_action_tracker(tmp_path):
    """Verify SQLite action tracker persistence."""
    test_db = str(tmp_path / "test_peoplepulse.db")
    init_db(test_db)
    
    add_retention_action(
        title="Test Initiative",
        pillar="Workload",
        department="Engineering",
        owner="Lead HR",
        issue="High Overtime",
        action="Workload redistribution",
        kpi_name="Overtime Hrs",
        kpi_base="12.0",
        kpi_target="5.0",
        review_date="2026-06-30",
        db_path=test_db
    )
    
    actions = get_all_actions(test_db)
    assert len(actions) >= 1
    action_id = int(actions.iloc[0]["id"])
    
    update_action_status(action_id, "In Progress", followup_kpi="6.2", notes="Underway", db_path=test_db)
    updated = get_all_actions(test_db)
    assert updated.iloc[0]["status"] == "In Progress"
    assert updated.iloc[0]["followup_kpi_val"] == "6.2"
    
    delete_retention_action(action_id, db_path=test_db)
    final_actions = get_all_actions(test_db)
    assert len(final_actions) == len(actions) - 1
