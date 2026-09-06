import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Attrition Report", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────
C_LEFT   = "#C0392B"
C_STAYED = "#2E6B9E"
C_ACCENT = "#E8956D"
C_GRID   = "#F0EDE8"
C_MID    = "#B8C9D9"
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
.page-title {{
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 600;
    color: {C_TEXT};
}}
.page-sub {{
    font-size: 13px;
    color: #8A847C;
    margin-bottom: 18px;
}}
.section-header {{
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 600;
    color: {C_TEXT};
    border-bottom: 2px solid {C_ACCENT};
}}
.insight {{
    background: #FFF8F4;
    border-left: 3px solid {C_ACCENT};
    padding: 9px 13px;
    font-size: 12.5px;
    color: #5A4A3A;
    margin-top: 5px;
}}
</style>
""", unsafe_allow_html=True)

def ins(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("cleaned_hr_data.csv")
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["AgeGroup"] = pd.cut(df["Age"], bins=[18,25,35,45,55,60],
                            labels=["18-25","26-35","36-45","46-55","56+"])
    df["TenureGroup"] = pd.cut(df["YearsAtCompany"], bins=[-1,2,5,10,20,100],
                               labels=["0-2 yrs","3-5 yrs","6-10 yrs","11-20 yrs","20+ yrs"])
    df["Status"] = df["Attrition"].map({0:"Stayed",1:"Left"})
    df["SatLabel"] = df["JobSatisfaction"].map({1:"Low",2:"Medium",3:"High",4:"Very High"})
    df["WLBLabel"] = df["WorkLifeBalance"].map({1:"Poor",2:"Fair",3:"Good",4:"Excellent"})
    return df

try:
    df = load()
except FileNotFoundError:
    up = st.sidebar.file_uploader("Upload HR CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
    else:
        st.warning("Please upload your HR dataset to continue.")
        st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### HR Attrition Report")
    page = st.radio("", ["Overview", "People & Roles", "Work Conditions"], label_visibility="collapsed")
    depts   = st.multiselect("Department", sorted(df["Department"].unique()), default=list(df["Department"].unique()))
    genders = st.multiselect("Gender", sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
    travel  = st.multiselect("Business Travel", sorted(df["BusinessTravel"].unique()), default=list(df["BusinessTravel"].unique()))

# ─────────────────────────────────────────────────────────────
# FILTERED DATA
# ─────────────────────────────────────────────────────────────
f = df[df["Department"].isin(depts) & df["Gender"].isin(genders) & df["BusinessTravel"].isin(travel)].copy()
total  = len(f)
n_left = int(f["Attrition"].sum())
n_stay = total - n_left
rate   = round(n_left / total * 100, 1) if total > 0 else 0

def attr_pct(col):
    t = (pd.crosstab(f[col], f["Status"], normalize="index") * 100).round(1).reset_index()
    if "Left" not in t.columns:
        t["Left"] = 0.0
    return t

# =====================================================================
# PAGE ROUTING
# =====================================================================
if page == "Overview":
    # --- Overview page content (single column layout) ---
    # (same as corrected block I gave earlier)
    ...

elif page == "People & Roles":
    # --- People & Roles page content (single column layout) ---
    ...

elif page == "Work Conditions":
    # --- Work Conditions page content (single column layout) ---
    ...
