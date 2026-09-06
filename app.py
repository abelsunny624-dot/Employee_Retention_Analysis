import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Report",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# PALETTE  — warm slate + terracotta accent
# ─────────────────────────────────────────────────────────────
C_LEFT   = "#C0392B"   # deep terracotta  — employees who left
C_STAYED = "#2E6B9E"   # slate blue       — employees who stayed
C_ACCENT = "#E8956D"   # soft amber       — highlight / secondary bars
C_GRID   = "#F0EDE8"   # warm off-white   — chart backgrounds
C_MID    = "#B8C9D9"   # muted blue-grey  — non-highlighted bars
C_TEXT   = "#2C2C2C"

CHART_FONT = "Georgia, serif"
BODY_FONT  = "Helvetica Neue, sans-serif"


def base_layout(height=340):
    return dict(
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C_GRID,
        font=dict(family=BODY_FONT, size=12, color=C_TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#D0CBC4"),
        yaxis=dict(showgrid=True, gridcolor="#DDD8D2", zeroline=False, linecolor="#D0CBC4"),
    )

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');
...
</style>
""", unsafe_allow_html=True)

def ins(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("cleaned_hr_data.csv")
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["AgeGroup"] = pd.cut(df["Age"],
                            bins=[18, 25, 35, 45, 55, 60],
                            labels=["18-25", "26-35", "36-45", "46-55", "56+"])
    df["TenureGroup"] = pd.cut(df["YearsAtCompany"],
                               bins=[-1, 2, 5, 10, 20, 100],
                               labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-20 yrs", "20+ yrs"])
    df["Status"] = df["Attrition"].map({0: "Stayed", 1: "Left"})
    df["SatLabel"] = df["JobSatisfaction"].map({1: "Low", 2: "Medium", 3: "High", 4: "Very High"})
    df["WLBLabel"]  = df["WorkLifeBalance"].map({1: "Poor", 2: "Fair", 3: "Good", 4: "Excellent"})
    return df

# Sidebar, filters, pages (Overview, People & Roles, Work Conditions) follow...
