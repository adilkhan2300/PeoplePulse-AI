"""
HR Action Tracker and SQLite Persistence Module for PeoplePulse AI
Stores, updates, and tracks evidence-based retention initiatives, baseline KPIs,
target timelines, and post-intervention evaluations in a local SQLite database.
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "peoplepulse.db"

def init_db(db_path=DB_PATH):
    """Initializes SQLite database and tables for retention action management."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retention_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            pillar TEXT NOT NULL,
            department TEXT NOT NULL,
            owner TEXT NOT NULL,
            identified_issue TEXT NOT NULL,
            proposed_action TEXT NOT NULL,
            baseline_kpi_name TEXT NOT NULL,
            baseline_kpi_val TEXT NOT NULL,
            target_kpi_val TEXT NOT NULL,
            followup_kpi_val TEXT,
            status TEXT NOT NULL,
            review_date TEXT NOT NULL,
            evaluation_notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    
    # Pre-populate sample benchmark action items if empty
    cursor.execute("SELECT COUNT(*) FROM retention_actions")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_actions = [
            (
                "Engineering 45-Hour Overtime Circuit Breaker",
                "Workload & Wellbeing",
                "Engineering",
                "VP of Engineering & HRBP",
                "Sustained overtime exceeding 10h/week during Q3 sprint cycle.",
                "Cap overtime at 8 hours/week, mandate comp days post sprint, and re-allocate 2 backend developers.",
                "Avg Weekly Overtime Hours",
                "9.8 hrs/wk",
                "< 4.0 hrs/wk",
                "4.5 hrs/wk",
                "In Progress",
                "2026-03-31",
                "Sprint velocity re-calibrated; initial team sentiment shows positive recovery.",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Mid-Year Tech IC Compa-Ratio Parity Review",
                "Compensation & Career Growth",
                "Engineering & Product",
                "Head of Total Rewards",
                "28% of Senior Engineers paid below 90% of market benchmark.",
                "Execute targeted $85,000 off-cycle equity & base salary alignment pool.",
                "Median Compa-Ratio",
                "0.86",
                "0.98",
                "",
                "Approved",
                "2026-04-15",
                "Awaiting Q1 budget release approval from CFO.",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Sales New Manager Enablement & Coaching",
                "Engagement & Leadership",
                "Sales",
                "Director of Talent Development",
                "Manager satisfaction dropped to 4.8/10 following recent team reorg.",
                "Conduct mandatory 4-week coaching program for 6 first-time sales managers.",
                "Manager Satisfaction Score",
                "4.8 / 10",
                "> 7.5 / 10",
                "",
                "Proposed",
                "2026-05-30",
                "Course curriculum finalized; cohort kicks off next month.",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        ]
        cursor.executemany("""
            INSERT INTO retention_actions (
                title, pillar, department, owner, identified_issue, proposed_action,
                baseline_kpi_name, baseline_kpi_val, target_kpi_val, followup_kpi_val,
                status, review_date, evaluation_notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_actions)
        conn.commit()
        
    conn.close()

def get_all_actions(db_path=DB_PATH):
    """Fetches all tracked retention actions as a Pandas DataFrame."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM retention_actions ORDER BY id DESC", conn)
    conn.close()
    return df

def add_retention_action(
    title, pillar, department, owner, issue, action,
    kpi_name, kpi_base, kpi_target, review_date, status="Proposed", notes="", db_path=DB_PATH
):
    """Inserts a new retention intervention action item."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO retention_actions (
            title, pillar, department, owner, identified_issue, proposed_action,
            baseline_kpi_name, baseline_kpi_val, target_kpi_val, followup_kpi_val,
            status, review_date, evaluation_notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        title, pillar, department, owner, issue, action,
        kpi_name, str(kpi_base), str(kpi_target), "",
        status, review_date, notes, now_str
    ))
    conn.commit()
    conn.close()
    return True

def update_action_status(action_id, new_status, followup_kpi=None, notes=None, db_path=DB_PATH):
    """Updates the status, evaluation notes, and follow-up KPI of an action."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updates = ["status = ?"]
    params = [new_status]
    
    if followup_kpi is not None:
        updates.append("followup_kpi_val = ?")
        params.append(str(followup_kpi))
    if notes is not None:
        updates.append("evaluation_notes = ?")
        params.append(str(notes))
        
    params.append(action_id)
    query = f"UPDATE retention_actions SET {', '.join(updates)} WHERE id = ?"
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    return True

def delete_retention_action(action_id, db_path=DB_PATH):
    """Deletes an action record from the SQLite database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM retention_actions WHERE id = ?", (action_id,))
    conn.commit()
    conn.close()
    return True
