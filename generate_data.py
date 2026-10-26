"""
Synthetic HR Dataset Generator for PeoplePulse AI
Generates realistic, interconnected HR tables for 1,200 employees with plausible
workplace relationships (overtime, compa-ratio, burnout, engagement, career stagnancy).
Seed is fixed for reproducibility.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_datasets(output_dir="data", n_employees=1200, seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Departments and Roles
    dept_roles = {
        "Engineering": [
            ("Software Engineer", "IC", 70000, 110000),
            ("Senior Software Engineer", "Senior", 110000, 150000),
            ("Lead Engineer", "Lead", 145000, 185000),
            ("DevOps Engineer", "IC", 75000, 120000),
            ("Engineering Manager", "Manager", 150000, 200000),
        ],
        "Sales": [
            ("Sales Representative", "IC", 50000, 75000),
            ("Account Executive", "Senior", 75000, 110000),
            ("Enterprise Account Exec", "Lead", 110000, 160000),
            ("Sales Manager", "Manager", 120000, 175000),
        ],
        "Product": [
            ("Associate Product Manager", "Junior", 60000, 85000),
            ("Product Manager", "IC", 85000, 125000),
            ("Senior Product Manager", "Senior", 125000, 165000),
            ("Director of Product", "Director", 165000, 220000),
        ],
        "Customer Success": [
            ("Support Specialist", "Junior", 40000, 60000),
            ("Customer Success Manager", "IC", 60000, 85000),
            ("Senior CSM", "Senior", 85000, 110000),
            ("CS Team Lead", "Manager", 90000, 125000),
        ],
        "Marketing": [
            ("Content Specialist", "Junior", 45000, 65000),
            ("Marketing Manager", "Manager", 80000, 115000),
            ("Growth Marketer", "IC", 65000, 95000),
            ("Brand Director", "Director", 140000, 190000),
        ],
        "Human Resources": [
            ("HR Coordinator", "Junior", 42000, 58000),
            ("Talent Acquisition Partner", "IC", 60000, 85000),
            ("HR Business Partner", "Senior", 85000, 120000),
            ("People Operations Manager", "Manager", 100000, 140000),
        ],
        "Finance": [
            ("Financial Analyst", "IC", 60000, 88000),
            ("Senior Accountant", "Senior", 80000, 115000),
            ("Finance Manager", "Manager", 115000, 160000),
        ]
    }
    
    locations = ["New York, NY", "San Francisco, CA", "Austin, TX", "Chicago, IL", "Remote"]
    employment_types = ["Full-Time", "Full-Time", "Full-Time", "Full-Time", "Part-Time", "Contract"]
    
    # Base employee pool
    employee_ids = [f"EMP-{1000 + i}" for i in range(n_employees)]
    
    depts = list(dept_roles.keys())
    dept_weights = [0.32, 0.22, 0.12, 0.14, 0.08, 0.06, 0.06]
    
    emp_records = []
    perf_records = []
    workload_records = []
    comp_records = []
    eng_records = []
    train_records = []
    
    base_date = datetime(2025, 12, 31)
    
    # Build employee attributes
    for emp_id in employee_ids:
        dept = np.random.choice(depts, p=dept_weights)
        role_tuple = dept_roles[dept][np.random.choice(len(dept_roles[dept]))]
        job_role, job_level, min_sal, max_sal = role_tuple
        loc = np.random.choice(locations, p=[0.25, 0.25, 0.20, 0.15, 0.15])
        emp_type = np.random.choice(employment_types)
        
        # Hire date between 6 months and 7 years ago
        tenure_days = np.random.randint(120, 2600)
        hire_date = base_date - timedelta(days=int(tenure_days))
        
        # Risk factors synthetic generation (hidden root causes)
        # Factors: High Overtime (Engineering/Sales), Low Compa-Ratio, Low Mgr Satisfaction, Stagnant Promotion
        is_high_ot_prone = dept in ["Engineering", "Sales", "Customer Success"] and np.random.rand() < 0.35
        is_underpaid = np.random.rand() < 0.28
        is_bad_manager = np.random.rand() < 0.22
        is_stagnant = tenure_days > 730 and np.random.rand() < 0.40
        
        # Overtime & Workload
        base_hours = 40.0
        if is_high_ot_prone:
            weekly_hours = np.random.normal(48, 5)
            ot_hours = max(0, weekly_hours - 40)
            workload_score = np.clip(np.random.normal(82, 10), 40, 100)
            leave_days = np.random.randint(3, 12)
            absenteeism_days = np.random.randint(2, 14)
        else:
            weekly_hours = np.random.normal(40, 3)
            ot_hours = max(0, weekly_hours - 40)
            workload_score = np.clip(np.random.normal(55, 12), 20, 90)
            leave_days = np.random.randint(8, 22)
            absenteeism_days = np.random.randint(0, 6)
            
        projects_assigned = max(1, int(np.random.poisson(3.5 if is_high_ot_prone else 2.2)))
        
        # Compensation
        mid_sal = (min_sal + max_sal) / 2
        if is_underpaid:
            base_sal = np.random.uniform(min_sal * 0.92, mid_sal * 0.95)
            salary_change_pct = np.random.uniform(0.0, 2.5)
            comp_band = "Below Benchmark"
        else:
            base_sal = np.random.uniform(mid_sal, max_sal * 1.05)
            salary_change_pct = np.random.uniform(3.0, 9.5)
            comp_band = "Market Competitive" if base_sal <= max_sal else "Above Market"
            
        bonus = base_sal * np.random.uniform(0.04, 0.18) if emp_type == "Full-Time" else 0.0
        promo_status = "Promoted in last 12m" if (not is_stagnant and np.random.rand() < 0.25) else "No recent promotion"
        
        # Performance
        perf_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.04, 0.12, 0.50, 0.26, 0.08])
        goal_completion = np.clip(perf_rating * 20 + np.random.normal(0, 7), 35, 100)
        mgr_feedback = np.clip(np.random.normal(6.5 if not is_bad_manager else 4.2, 1.5), 1, 10)
        
        # Engagement
        if is_bad_manager:
            mgr_sat = np.clip(np.random.normal(4.0, 1.4), 1, 10)
        else:
            mgr_sat = np.clip(np.random.normal(7.8, 1.3), 1, 10)
            
        if is_high_ot_prone:
            wlb_score = np.clip(np.random.normal(4.2, 1.5), 1, 10)
        else:
            wlb_score = np.clip(np.random.normal(7.6, 1.4), 1, 10)
            
        job_sat = np.clip((mgr_sat * 0.3 + wlb_score * 0.3 + (1 if not is_underpaid else -1) * 1.5 + np.random.normal(4.5, 1.2)), 1, 10)
        recognition_score = np.clip(np.random.normal(6.8 if not is_stagnant else 4.5, 1.6), 1, 10)
        eng_score = np.clip(np.mean([job_sat, mgr_sat, wlb_score, recognition_score]) * 10, 15, 100)
        intent_to_stay = "Yes" if eng_score > 65 else ("Undecided" if eng_score > 48 else "No")
        
        # Training
        training_hours = max(0, int(np.random.normal(24 if not is_stagnant else 8, 10)))
        skills_score = np.clip(training_hours * 2.2 + np.random.normal(30, 10), 10, 100)
        train_program = np.random.choice(["Leadership Accelerate", "Technical Upskilling", "Compliance & Safety", "None"], p=[0.25, 0.40, 0.25, 0.10])
        train_completed = "Completed" if training_hours > 10 else "Incomplete"
        
        # Attrition likelihood formula (ground truth generating function)
        # Probability based on verified organizational risk drivers
        logit = -2.9 \
            + 0.08 * ot_hours \
            + 0.04 * (80 - eng_score) \
            + 0.06 * absenteeism_days \
            + (0.85 if is_underpaid else -0.3) \
            + (0.90 if is_bad_manager else -0.2) \
            + (0.65 if is_stagnant else -0.1) \
            + (-0.5 if perf_rating >= 4 else 0.3)
            
        prob_attrition = 1 / (1 + np.exp(-logit))
        
        # Determine exit status
        has_exited = np.random.rand() < prob_attrition
        
        if has_exited:
            # Termination date within last 12 months
            days_ago = np.random.randint(15, 360)
            term_date = base_date - timedelta(days=days_ago)
            # Ensure term_date is after hire_date
            if term_date <= hire_date:
                term_date = hire_date + timedelta(days=90)
            
            # Voluntary vs Involuntary
            if perf_rating <= 2 and np.random.rand() < 0.6:
                term_type = "Involuntary"
                term_reason = np.random.choice(["Performance", "Restructuring", "Attendance Violation"])
            else:
                term_type = "Voluntary"
                if is_high_ot_prone and wlb_score < 5:
                    term_reason = "Burnout / Work-Life Balance"
                elif is_underpaid:
                    term_reason = "Better Compensation Offer"
                elif is_bad_manager:
                    term_reason = "Relationship with Manager"
                elif is_stagnant:
                    term_reason = "Lack of Career Growth"
                else:
                    term_reason = np.random.choice(["Personal Reasons", "Relocation", "Career Pivot"])
            emp_status = "Terminated"
        else:
            term_date = None
            term_type = None
            term_reason = None
            emp_status = "Active"
            
        # Append records
        emp_records.append({
            "employee_id": emp_id,
            "department": dept,
            "job_role": job_role,
            "job_level": job_level,
            "location": loc,
            "employment_type": emp_type,
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "employment_status": emp_status,
            "termination_date": term_date.strftime("%Y-%m-%d") if term_date else "",
            "termination_type": term_type if term_type else "",
            "termination_reason": term_reason if term_reason else ""
        })
        
        workload_records.append({
            "employee_id": emp_id,
            "reporting_period": "2025-Q4",
            "weekly_working_hours": round(weekly_hours, 1),
            "overtime_hours": round(ot_hours, 1),
            "projects_assigned": projects_assigned,
            "workload_score": round(workload_score, 1),
            "leave_days": leave_days,
            "absenteeism_days": absenteeism_days
        })
        
        comp_records.append({
            "employee_id": emp_id,
            "reporting_period": "2025-Q4",
            "base_salary": round(base_sal, 2),
            "bonus": round(bonus, 2),
            "salary_change_percentage": round(salary_change_pct, 2),
            "compensation_band": comp_band,
            "promotion_status": promo_status
        })
        
        perf_records.append({
            "employee_id": emp_id,
            "review_date": "2025-11-15",
            "performance_rating": perf_rating,
            "goal_completion_percentage": round(goal_completion, 1),
            "promotion_history": "Promoted" if promo_status == "Promoted in last 12m" else "Standard",
            "manager_feedback_score": round(mgr_feedback, 1)
        })
        
        eng_records.append({
            "employee_id": emp_id,
            "survey_date": "2025-10-20",
            "engagement_score": round(eng_score, 1),
            "job_satisfaction": round(job_sat, 1),
            "manager_satisfaction": round(mgr_sat, 1),
            "work_life_balance_score": round(wlb_score, 1),
            "recognition_score": round(recognition_score, 1),
            "intent_to_stay": intent_to_stay
        })
        
        train_records.append({
            "employee_id": emp_id,
            "training_date": "2025-09-10",
            "training_hours": training_hours,
            "training_program": train_program,
            "training_completion_status": train_completed,
            "skills_development_score": round(skills_score, 1)
        })
        
    # Recruitment costs benchmark table
    recruitment_cost_records = []
    for dept, roles in dept_roles.items():
        for role_tuple in roles:
            role_name, level, min_s, max_s = role_tuple
            mid = (min_s + max_s) / 2
            # Typical replacement cost: 25% - 40% of base salary
            rec_cost = round(mid * 0.18, 2)
            onboard_cost = round(mid * 0.08, 2)
            train_cost = round(mid * 0.06, 2)
            time_to_fill = int(np.random.uniform(30, 75))
            
            recruitment_cost_records.append({
                "department": dept,
                "job_role": role_name,
                "recruitment_cost": rec_cost,
                "onboarding_cost": onboard_cost,
                "training_cost": train_cost,
                "average_time_to_fill_days": time_to_fill
            })
            
    # Save datasets
    df_emp = pd.DataFrame(emp_records)
    df_workload = pd.DataFrame(workload_records)
    df_comp = pd.DataFrame(comp_records)
    df_perf = pd.DataFrame(perf_records)
    df_eng = pd.DataFrame(eng_records)
    df_train = pd.DataFrame(train_records)
    df_costs = pd.DataFrame(recruitment_cost_records)
    
    df_emp.to_csv(os.path.join(output_dir, "employees.csv"), index=False)
    df_workload.to_csv(os.path.join(output_dir, "workload.csv"), index=False)
    df_comp.to_csv(os.path.join(output_dir, "compensation.csv"), index=False)
    df_perf.to_csv(os.path.join(output_dir, "performance.csv"), index=False)
    df_eng.to_csv(os.path.join(output_dir, "engagement.csv"), index=False)
    df_train.to_csv(os.path.join(output_dir, "training.csv"), index=False)
    df_costs.to_csv(os.path.join(output_dir, "recruitment_costs.csv"), index=False)
    
    print(f"Successfully generated {len(df_emp)} employee records across 7 HR tables in '{output_dir}'.")
    exits = df_emp[df_emp['employment_status'] == 'Terminated']
    print(f"Total employees: {len(df_emp)}, Exited: {len(exits)} ({len(exits)/len(df_emp)*100:.1f}%)")

if __name__ == "__main__":
    generate_datasets("data", 1200, 42)
