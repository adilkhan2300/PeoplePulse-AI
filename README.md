# PeoplePulse AI — Strategic Attrition & Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **PeoplePulse AI** is an enterprise-grade, evidence-based workforce intelligence and employee retention platform. It unites rigorous statistical root-cause identification, calibrated machine learning flight risk prediction, granular replacement cost financial modeling, interactive what-if retention scenario simulation, and an integrated SQLite action tracker.

---

## 🌟 Key Highlights & Capabilities

1. **Multi-Table Relational HR Architecture**:
   - Ingests and seamlessly joins 7 standard HR schemas on `employee_id` and role benchmarks: `employees`, `compensation`, `engagement`, `performance`, `recruitment_costs`, `training`, and `workload`.
   - Built-in synthetic dataset generator modeling 1,200 employees with realistic organizational dynamics (burnout, compa-ratio deficits, management sentiment divergence).

2. **Executive Workforce Intelligence**:
   - Computes standard HR analytics metrics adhering to formal workforce accounting formulas:
     $$\text{Attrition Rate} = \frac{\text{Exited Employees}}{\text{Active Employees} + \frac{\text{Exited Employees}}{2}} \times 100$$
   - Departmental benchmarks, tenure cohort survival stay-curves, and departure velocity tracking.

3. **Rigorous Statistical Root-Cause Analysis**:
   - Independent two-sample Welch's t-tests, non-parametric Mann-Whitney U tests, and Cohen's $d$ effect sizes.
   - Categorical Pearson Chi-Square tests of independence with Cramér's $V$.
   - Adjusted multivariable logistic regression yielding Adjusted Odds Ratios ($\text{OR} = \exp(\beta)$) with 95% Confidence Intervals and $p$-values.

4. **Calibrated Machine Learning Flight Risk Engine**:
   - Supervised models: Logistic Regression (Baseline), Random Forest (Nonlinear), and Gradient Boosting (Ensemble).
   - Evaluated using ROC-AUC, PR-AUC, Precision, Recall, F1-Score, and Brier Score.
   - Model-agnostic Permutation Feature Importance & SHAP drivers.
   - Dynamic classification threshold tuning with precision/recall trade-off matrix.
   - Active employee risk categorization: **Elevated Risk** ($\ge 60\%$), **Moderate Risk** ($35\text{--}59\%$), and **Low Risk** ($<35\%$).

5. **Turnover Financial Drag & ROI Simulator**:
   - Replacement cost modeling: Direct Recruitment, Onboarding, Replacement Training, Vacancy Drag ($\text{Days to Fill} \times \text{Daily Cost}$), and Productivity Ramp-Up drag.
   - Interactive what-if scenario simulator evaluating overtime caps, compa-ratio alignment pools, cultural engagement lifts, and training expansions.

6. **Retention Playbook & Action Tracker**:
   - Evidence-based intervention cards mapped to 4 pillars: *Workload & Wellbeing*, *Compensation & Career Growth*, *Engagement & Leadership*, and *Learning & Development*.
   - Integrated SQLite database (`peoplepulse.db`) with full CRUD support for organizational retention initiatives.

7. **Multi-Format Executive Reporting**:
   - Board-level executive PDF dossiers generated via ReportLab with typography and AI disclosures.
   - Multi-sheet Excel workbooks (`.xlsx`) via openpyxl.

---

## 📂 Project Structure

```
PB5/
├── .streamlit/
│   └── config.toml                  # Dark theme styling and server config
├── data/                            # 7 standard HR relational CSV datasets
│   ├── compensation.csv
│   ├── employees.csv
│   ├── engagement.csv
│   ├── performance.csv
│   ├── recruitment_costs.csv
│   ├── training.csv
│   └── workload.csv
├── modules/                         # Core analytics & machine learning engine
│   ├── __init__.py
│   ├── action_tracker.py            # SQLite retention action persistence
│   ├── attrition_analysis.py        # Executive KPIs, cohorts, and trends
│   ├── cost_analysis.py             # Replacement cost financial modeling
│   ├── data_loader.py               # Ingestion, joins, and template exports
│   ├── data_validation.py           # Schema verification and quality checks
│   ├── explainability.py            # Permutation importance & SHAP attribution
│   ├── feature_engineering.py       # Compa-ratio, burnout index, tenure cohorts
│   ├── llm_insights.py              # LLM executive briefing & deterministic engine
│   ├── prediction_models.py         # Supervised ML pipelines & flight risk scoring
│   ├── reporting.py                 # ReportLab PDF & openpyxl Excel exports
│   ├── retention_recommendations.py # 4-pillar evidence-based playbooks
│   ├── scenario_simulator.py        # What-if ROI and attrition reduction math
│   └── statistical_analysis.py      # t-tests, Mann-Whitney U, Odds Ratios, Chi-Square
├── tests/
│   └── test_peoplepulse.py          # Comprehensive pytest test suite
├── utils/
│   ├── __init__.py
│   └── theme.py                     # Custom CSS, KPI cards, and Plotly templates
├── .env.example                     # Environment template (OpenAI API key)
├── .gitignore                       # Git ignore specifications
├── app.py                           # Main Streamlit web application
├── generate_data.py                 # Synthetic dataset generator
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
# Navigate to the workspace directory
cd PB5

# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Verify with Unit Tests
```bash
python -m pytest
```
All 7 unit tests validate data ingestion, feature engineering, attrition KPI math, statistical tests, machine learning pipelines, turnover costs, and SQLite persistence.

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Core Methodology & Mathematical Formulations

### 1. Average Headcount & Attrition Rate
$$\text{Average Headcount} = \text{Active Employees} + \frac{\text{Exits}}{2}$$
$$\text{Attrition Rate (\%)} = \frac{\text{Exits}}{\text{Average Headcount}} \times 100$$

### 2. Cohen's $d$ Effect Size
$$d = \frac{\bar{X}_{\text{exited}} - \bar{X}_{\text{active}}}{s_{\text{pooled}}}, \quad \text{where } s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

### 3. Adjusted Odds Ratio (Logistic Regression)
$$\text{OR} = \exp(\beta_j), \quad 95\%\text{ CI} = \left[\exp(\beta_j - 1.96 \cdot \text{SE}), \, \exp(\beta_j + 1.96 \cdot \text{SE})\right]$$

### 4. Turnover Financial Replacement Cost
$$\text{Cost} = \text{Recruitment} + \text{Onboarding} + \text{Training} + (\text{Vacancy Days} \times \text{Daily Coverage Drag}) + \text{Productivity Ramp Loss}$$

---

## 🛡️ Responsible AI Disclosures
PeoplePulse AI is designed strictly as an aggregate decision-support and systemic organizational policy platform. It is **not** intended for adverse individual employment actions. All predictive risk scores require qualitative contextualization by qualified HR professionals.

---

## 📄 License
This project is licensed under the MIT License.
