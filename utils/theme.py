"""
Theme and UI helper functions for PeoplePulse AI
Provides consistent styling, cards, alert banners, and Plotly templates.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PRIMARY_COLOR = "#3B82F6"     # Bright Blue
SECONDARY_COLOR = "#6366F1"   # Indigo
SUCCESS_COLOR = "#10B981"     # Emerald
WARNING_COLOR = "#F59E0B"     # Amber
DANGER_COLOR = "#EF4444"      # Rose / Red
BG_CARD = "#1E293B"           # Slate 800
TEXT_MAIN = "#F8FAFC"         # Slate 50
TEXT_MUTED = "#94A3B8"        # Slate 400

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .hero-title {
        color: #F8FAFC;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Metric Card */
    .kpi-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
    }
    
    /* Risk Badge */
    .badge-elevated {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11px;
    }
    
    /* Callout Alert */
    .callout-box {
        background: #0F172A;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 14px;
        margin-bottom: 14px;
        color: #E2E8F0;
        font-size: 13.5px;
    }
    .callout-box.warning {
        border-left-color: #F59E0B;
        background: #18181B;
    }
    .callout-box.danger {
        border-left-color: #EF4444;
        background: #18181B;
    }
    .callout-box.success {
        border-left-color: #10B981;
        background: #18181B;
    }
</style>
"""

def apply_app_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_hero(title, subtitle, badge_text="PEOPLEPULSE AI v2.5"):
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="hero-title">{title}</div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60A5FA; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;">
                {badge_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(col, title, value, subtext="", trend=""):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext} <span style="font-weight: 600; color: #38BDF8;">{trend}</span></div>
        </div>
        """, unsafe_allow_html=True)

def render_callout(text, kind="info"):
    icon = "ℹ️"
    if kind == "warning":
        icon = "⚠️"
    elif kind == "danger":
        icon = "🚨"
    elif kind == "success":
        icon = "✅"
    st.markdown(f"""
    <div class="callout-box {kind}">
        <strong>{icon} </strong> {text}
    </div>
    """, unsafe_allow_html=True)

def get_plotly_layout():
    """Returns a sleek, dark-slate Plotly layout for charts."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        plot_bgcolor="rgba(15, 23, 42, 0.0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#E2E8F0"),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.07)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.07)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
