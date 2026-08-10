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

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: #FAF8F5;
    color: {C_TEXT};
}}
[data-testid="stSidebar"] {{
    background-color: #F2EFE9;
    border-right: 1px solid #E0DAD2;
}}
[data-testid="stSidebar"] * {{ font-family: 'DM Sans', sans-serif !important; }}

[data-testid="metric-container"] {{
    background: #FFFFFF;
    border: 1px solid #E8E2DA;
    border-radius: 8px;
    padding: 18px 20px !important;
}}
[data-testid="metric-container"] label {{
    font-size: 11px !important;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8A847C !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Playfair Display', serif;
    font-size: 30px !important;
    color: {C_TEXT} !important;
    font-weight: 600;
}}

.section-header {{
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 600;
    color: {C_TEXT};
    margin: 20px 0 4px 0;
    padding-bottom: 5px;
    border-bottom: 2px solid {C_ACCENT};
    display: inline-block;
}}
.insight {{
    background: #FFF8F4;
    border-left: 3px solid {C_ACCENT};
    padding: 9px 13px;
    border-radius: 0 6px 6px 0;
    font-size: 12.5px;
    color: #5A4A3A;
    line-height: 1.55;
    margin-top: 5px;
}}
.page-title {{
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 600;
    color: {C_TEXT};
    margin-bottom: 2px;
}}
.page-sub {{
    font-size: 13px;
    color: #8A847C;
    font-weight: 300;
    margin-bottom: 18px;
}}
hr {{ border: none; border-top: 1px solid #E8E2DA; margin: 24px 0; }}
[data-testid="stDataFrame"] {{
    border: 1px solid #E8E2DA;
    border-radius: 8px;
    overflow: hidden;
}}
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

try:
    df = load()
except FileNotFoundError:
    up = st.sidebar.file_uploader("Upload HR CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        if df["Attrition"].dtype == object:
            df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
        df["AgeGroup"]    = pd.cut(df["Age"], bins=[18,25,35,45,55,60],
                                   labels=["18-25","26-35","36-45","46-55","56+"])
        df["TenureGroup"] = pd.cut(df["YearsAtCompany"], bins=[-1,2,5,10,20,100],
                                   labels=["0-2 yrs","3-5 yrs","6-10 yrs","11-20 yrs","20+ yrs"])
        df["Status"]   = df["Attrition"].map({0:"Stayed",1:"Left"})
        df["SatLabel"] = df["JobSatisfaction"].map({1:"Low",2:"Medium",3:"High",4:"Very High"})
        df["WLBLabel"] = df["WorkLifeBalance"].map({1:"Poor",2:"Fair",3:"Good",4:"Excellent"})
    else:
        st.warning("Please upload your HR dataset to continue.")
        st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### HR Attrition Report")
    st.markdown("---")
    st.markdown("**Page**")
    page = st.radio("", ["Overview", "People & Roles", "Work Conditions"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Filters**")
    depts   = st.multiselect("Department",
                             sorted(df["Department"].unique()),
                             default=list(df["Department"].unique()))
    genders = st.multiselect("Gender",
                             sorted(df["Gender"].unique()),
                             default=list(df["Gender"].unique()))
    travel  = st.multiselect("Business Travel",
                             sorted(df["BusinessTravel"].unique()),
                             default=list(df["BusinessTravel"].unique()))
    st.markdown("---")
    st.caption("IBM HR Analytics Dataset")

# ─────────────────────────────────────────────────────────────
# FILTERED DATA
# ─────────────────────────────────────────────────────────────
f = df[
    df["Department"].isin(depts) &
    df["Gender"].isin(genders) &
    df["BusinessTravel"].isin(travel)
].copy()

total  = len(f)
n_left = int(f["Attrition"].sum())
n_stay = total - n_left
rate   = round(n_left / total * 100, 1) if total > 0 else 0

# ─────────────────────────────────────────────────────────────
# helper: attrition % crosstab
# ─────────────────────────────────────────────────────────────
def attr_pct(col):
    t = (pd.crosstab(f[col], f["Status"], normalize="index") * 100).round(1).reset_index()
    t.columns.name = None
    if "Left" not in t.columns:
        t["Left"] = 0.0
    return t


# =====================================================================
# PAGE 1 — OVERVIEW
# =====================================================================
if page == "Overview":

    st.markdown('<p class="page-title">Attrition Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">A high-level summary of employee retention across the organisation.</p>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees",    f"{total:,}")
    c2.metric("Employees Who Left", f"{n_left:,}")
    c3.metric("Employees Retained", f"{n_stay:,}")
    if rate >= 20:
        c4.metric("Attrition Rate", f"{rate}%", delta="Above target", delta_color="inverse")
    elif rate >= 10:
        c4.metric("Attrition Rate", f"{rate}%", delta="Moderate — monitor closely", delta_color="off")
    else:
        c4.metric("Attrition Rate", f"{rate}%", delta="Within healthy range", delta_color="normal")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row 1: Donut  +  Department bar
    col1, col2 = st.columns([1, 1.6])

    with col1:
        section("Retention Split")
        fig = go.Figure(go.Pie(
            labels=["Left", "Stayed"],
            values=[n_left, n_stay],
            hole=0.62,
            marker=dict(colors=[C_LEFT, C_STAYED]),
            textinfo="percent",
            textfont=dict(size=13, family=BODY_FONT),
            hovertemplate="%{label}: %{value:,} employees<extra></extra>"
        ))
        fig.add_annotation(text=f"<b>{rate}%</b><br>Left",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, family=CHART_FONT, color=C_TEXT))
        lo = base_layout(height=290)
        lo["showlegend"] = True
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ins(f"{rate}% of employees have left. Industry benchmark is typically 10-15%.")

    with col2:
        section("Attrition Rate by Department")
        dept = (f.groupby("Department")["Attrition"]
                .mean().mul(100).round(1).reset_index()
                .rename(columns={"Attrition": "Rate"})
                .sort_values("Rate", ascending=True))
        fig = go.Figure(go.Bar(
            x=dept["Rate"], y=dept["Department"], orientation="h",
            marker=dict(color=[C_LEFT if v == dept["Rate"].max() else C_ACCENT for v in dept["Rate"]],
                        line=dict(width=0)),
            text=dept["Rate"].map(lambda v: f"{v}%"),
            textposition="outside",
            hovertemplate="%{y}: %{x}%<extra></extra>"
        ))
        lo = base_layout(height=290)
        lo["xaxis"]["range"]   = [0, dept["Rate"].max() + 8]
        lo["xaxis"]["showgrid"] = False
        lo["yaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        top_dept = dept.iloc[-1]
        ins(f"The {top_dept['Department']} department has the highest attrition at {top_dept['Rate']}%. Prioritise retention initiatives there.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row 2: Summary table  +  Gender
    col3, col4 = st.columns([1.6, 1])

    with col3:
        section("Department Summary")
        snap = (f.groupby("Department")
                .agg(Total=("Attrition","count"), Left=("Attrition","sum"))
                .assign(**{"Retention Rate (%)": lambda d: ((d["Total"]-d["Left"])/d["Total"]*100).round(1),
                           "Attrition Rate (%)": lambda d: (d["Left"]/d["Total"]*100).round(1)})
                .reset_index()
                .rename(columns={"Total": "Total Employees", "Left": "Employees Left"}))
        st.dataframe(snap, use_container_width=True, hide_index=True)

    with col4:
        section("Attrition by Gender")
        gen = attr_pct("Gender")
        stayed_col = gen.get("Stayed", pd.Series([0]*len(gen)))
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Left",   x=gen["Gender"], y=gen["Left"],
                             marker_color=C_LEFT,   text=gen["Left"].map(lambda v: f"{v:.0f}%"),
                             textposition="auto"))
        fig.add_trace(go.Bar(name="Stayed", x=gen["Gender"], y=stayed_col,
                             marker_color=C_STAYED, text=stayed_col.map(lambda v: f"{v:.0f}%"),
                             textposition="auto"))
        lo = base_layout(height=270)
        lo["barmode"]        = "stack"
        lo["yaxis"]["title"] = "% of Gender Group"
        lo["plot_bgcolor"]   = C_GRID
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =====================================================================
# PAGE 2 — PEOPLE & ROLES
# =====================================================================
elif page == "People & Roles":

    st.markdown('<p class="page-title">People & Roles</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Which roles, age groups and tenure bands are most at risk?</p>', unsafe_allow_html=True)

    # Row 1: Age  +  Job role
    col1, col2 = st.columns(2)

    with col1:
        section("Attrition Rate by Age Group")
        age = attr_pct("AgeGroup")
        age["AgeGroup"] = age["AgeGroup"].astype(str)
        fig = go.Figure(go.Bar(
            x=age["AgeGroup"], y=age["Left"],
            marker_color=[C_LEFT if v == age["Left"].max() else C_ACCENT for v in age["Left"]],
            text=age["Left"].map(lambda v: f"{v:.0f}%"),
            textposition="outside",
            hovertemplate="%{x}: %{y}% left<extra></extra>"
        ))
        lo = base_layout(height=310)
        lo["yaxis"]["range"] = [0, age["Left"].max() + 12]
        lo["yaxis"]["title"] = "% who left"
        lo["xaxis"]["title"] = "Age Group"
        lo["plot_bgcolor"]   = C_GRID
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        top_age = age.loc[age["Left"].idxmax(), "AgeGroup"]
        ins(f"Employees aged {top_age} have the highest attrition. This group often moves for better career opportunities or compensation.")

    with col2:
        section("Attrition Rate by Job Role")
        role = (f.groupby("JobRole")["Attrition"]
                .mean().mul(100).round(1).reset_index()
                .rename(columns={"Attrition": "Rate"})
                .sort_values("Rate", ascending=True))
        fig = go.Figure(go.Bar(
            x=role["Rate"], y=role["JobRole"], orientation="h",
            marker=dict(color=[C_LEFT if v == role["Rate"].max() else C_MID for v in role["Rate"]],
                        line=dict(width=0)),
            text=role["Rate"].map(lambda v: f"{v}%"),
            textposition="outside",
            hovertemplate="%{y}: %{x}%<extra></extra>"
        ))
        lo = base_layout(height=310)
        lo["xaxis"]["range"]    = [0, role["Rate"].max() + 10]
        lo["xaxis"]["showgrid"] = False
        lo["yaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        top_role = role.iloc[-1]
        ins(f"{top_role['JobRole']} has the highest attrition at {top_role['Rate']}%, highlighted in red.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row 2: Tenure  +  Promotion gap
    col3, col4 = st.columns(2)

    with col3:
        section("Attrition by Years at Company")
        ten = attr_pct("TenureGroup")
        ten["TenureGroup"] = ten["TenureGroup"].astype(str)
        fig = go.Figure(go.Bar(
            x=ten["TenureGroup"], y=ten["Left"],
            marker_color=[C_LEFT if v == ten["Left"].max() else C_ACCENT for v in ten["Left"]],
            text=ten["Left"].map(lambda v: f"{v:.0f}%"),
            textposition="outside"
        ))
        lo = base_layout(height=300)
        lo["yaxis"]["range"] = [0, ten["Left"].max() + 12]
        lo["yaxis"]["title"] = "% who left"
        lo["xaxis"]["title"] = "Years at Company"
        lo["plot_bgcolor"]   = C_GRID
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        top_ten = ten.loc[ten["Left"].idxmax(), "TenureGroup"]
        ins(f"Employees in their {top_ten} are most likely to leave — the early-tenure risk window. Structured onboarding helps close this gap.")

    with col4:
        section("Avg Years Since Last Promotion")
        promo = f.groupby("Status")["YearsSinceLastPromotion"].mean().round(1).reset_index()
        fig = go.Figure(go.Bar(
            x=promo["Status"], y=promo["YearsSinceLastPromotion"],
            marker_color=[C_LEFT if s == "Left" else C_STAYED for s in promo["Status"]],
            text=promo["YearsSinceLastPromotion"].map(lambda v: f"{v} yrs"),
            textposition="outside",
            width=0.4
        ))
        lo = base_layout(height=300)
        lo["yaxis"]["range"] = [0, promo["YearsSinceLastPromotion"].max() + 1.5]
        lo["yaxis"]["title"] = "Avg years since last promotion"
        lo["xaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        avg_l = promo.loc[promo["Status"] == "Left",   "YearsSinceLastPromotion"].values[0]
        avg_s = promo.loc[promo["Status"] == "Stayed", "YearsSinceLastPromotion"].values[0]
        ins(f"Employees who left waited {avg_l:.1f} yrs since their last promotion vs {avg_s:.1f} yrs for those who stayed. Career stagnation is a clear signal.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row 3: Marital status
    section("Attrition Rate by Marital Status")
    ms = attr_pct("MaritalStatus").sort_values("Left", ascending=False)
    fig = go.Figure(go.Bar(
        x=ms["MaritalStatus"], y=ms["Left"],
        marker_color=[C_LEFT if v == ms["Left"].max() else C_MID for v in ms["Left"]],
        text=ms["Left"].map(lambda v: f"{v:.0f}%"),
        textposition="outside",
        width=0.35
    ))
    lo = base_layout(height=250)
    lo["yaxis"]["range"]    = [0, ms["Left"].max() + 10]
    lo["yaxis"]["title"]    = "% who left"
    lo["xaxis"]["showgrid"] = False
    lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ins("Single employees have the highest attrition — they have more flexibility to explore new opportunities. Consider mentoring and career development programmes for this group.")


# =====================================================================
# PAGE 3 — WORK CONDITIONS
# =====================================================================
elif page == "Work Conditions":

    st.markdown('<p class="page-title">Work Conditions</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">How pay, overtime, satisfaction and balance relate to who leaves.</p>', unsafe_allow_html=True)

    # Row 1: Salary  +  Overtime
    col1, col2 = st.columns(2)

    with col1:
        section("Average Monthly Pay — Stayed vs Left")
        sal = f.groupby("Status")["MonthlyIncome"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=sal["Status"], y=sal["MonthlyIncome"].round(0),
            marker_color=[C_LEFT if s == "Left" else C_STAYED for s in sal["Status"]],
            text=sal["MonthlyIncome"].round(0).map(lambda v: f"${int(v):,}"),
            textposition="outside",
            width=0.4
        ))
        lo = base_layout(height=300)
        lo["yaxis"]["range"]    = [0, sal["MonthlyIncome"].max() + 1200]
        lo["yaxis"]["title"]    = "Avg Monthly Income"
        lo["xaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        l_sal = sal.loc[sal["Status"] == "Left",   "MonthlyIncome"].values[0]
        s_sal = sal.loc[sal["Status"] == "Stayed", "MonthlyIncome"].values[0]
        ins(f"Employees who left earned ${int(l_sal):,}/month on average — ${int(s_sal-l_sal):,} less than those who stayed. Pay competitiveness is a key retention lever.")

    with col2:
        section("Overtime vs Attrition Rate")
        ot = attr_pct("OverTime")
        fig = go.Figure(go.Bar(
            x=ot["OverTime"], y=ot["Left"],
            marker_color=[C_LEFT if v == ot["Left"].max() else C_STAYED for v in ot["Left"]],
            text=ot["Left"].map(lambda v: f"{v:.0f}%"),
            textposition="outside",
            width=0.35
        ))
        lo = base_layout(height=300)
        lo["yaxis"]["range"]    = [0, ot["Left"].max() + 12]
        lo["yaxis"]["title"]    = "% who left"
        lo["xaxis"]["title"]    = "Works Overtime?"
        lo["xaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Yes" in ot["OverTime"].values and "No" in ot["OverTime"].values:
            ot_yes = ot.loc[ot["OverTime"] == "Yes", "Left"].values[0]
            ot_no  = ot.loc[ot["OverTime"] == "No",  "Left"].values[0]
            ratio  = round(ot_yes / ot_no, 1) if ot_no else "n/a"
            ins(f"{ot_yes:.0f}% of overtime workers leave vs {ot_no:.0f}% of non-overtime workers — {ratio}x higher. Workload is a critical driver worth addressing.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row 2: Job satisfaction  +  Work-life balance
    col3, col4 = st.columns(2)

    with col3:
        section("Attrition Rate by Job Satisfaction")
        sat_order = ["Low", "Medium", "High", "Very High"]
        sat = attr_pct("SatLabel")
        sat = sat[sat["SatLabel"].isin(sat_order)].copy()
        sat["SatLabel"] = pd.Categorical(sat["SatLabel"], categories=sat_order, ordered=True)
        sat = sat.sort_values("SatLabel")
        fig = go.Figure(go.Bar(
            x=sat["SatLabel"].astype(str), y=sat["Left"],
            marker_color=[C_LEFT if v == sat["Left"].max() else C_ACCENT for v in sat["Left"]],
            text=sat["Left"].map(lambda v: f"{v:.0f}%"),
            textposition="outside"
        ))
        lo = base_layout(height=300)
        lo["yaxis"]["range"] = [0, sat["Left"].max() + 10]
        lo["yaxis"]["title"] = "% who left"
        lo["xaxis"]["title"] = "Job Satisfaction"
        lo["plot_bgcolor"]   = C_GRID
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ins("Employees reporting low satisfaction leave at the highest rate. Even modest improvements in engagement can meaningfully cut turnover.")

    with col4:
        section("Attrition Rate by Work-Life Balance")
        wlb_order = ["Poor", "Fair", "Good", "Excellent"]
        wlb = attr_pct("WLBLabel")
        wlb = wlb[wlb["WLBLabel"].isin(wlb_order)].copy()
        wlb["WLBLabel"] = pd.Categorical(wlb["WLBLabel"], categories=wlb_order, ordered=True)
        wlb = wlb.sort_values("WLBLabel")
        fig = go.Figure(go.Bar(
            x=wlb["WLBLabel"].astype(str), y=wlb["Left"],
            marker_color=[C_LEFT if v == wlb["Left"].max() else C_ACCENT for v in wlb["Left"]],
            text=wlb["Left"].map(lambda v: f"{v:.0f}%"),
            textposition="outside"
        ))
        lo = base_layout(height=300)
        lo["yaxis"]["range"] = [0, wlb["Left"].max() + 10]
        lo["yaxis"]["title"] = "% who left"
        lo["xaxis"]["title"] = "Work-Life Balance"
        lo["plot_bgcolor"]   = C_GRID
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ins("Poor work-life balance is consistently tied to higher attrition. Flexible working policies can directly address this.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row 3: Commute  +  Travel frequency
    col5, col6 = st.columns(2)

    with col5:
        section("Average Commute Distance — Stayed vs Left")
        dist = f.groupby("Status")["DistanceFromHome"].mean().round(1).reset_index()
        fig = go.Figure(go.Bar(
            x=dist["Status"], y=dist["DistanceFromHome"],
            marker_color=[C_LEFT if s == "Left" else C_STAYED for s in dist["Status"]],
            text=dist["DistanceFromHome"].map(lambda v: f"{v} km"),
            textposition="outside",
            width=0.4
        ))
        lo = base_layout(height=270)
        lo["yaxis"]["range"]    = [0, dist["DistanceFromHome"].max() + 2]
        lo["yaxis"]["title"]    = "Avg Distance (km)"
        lo["xaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        d_l = dist.loc[dist["Status"] == "Left",   "DistanceFromHome"].values[0]
        d_s = dist.loc[dist["Status"] == "Stayed", "DistanceFromHome"].values[0]
        ins(f"Employees who left lived {d_l} km from the office vs {d_s} km for those who stayed. Long commutes erode work-life quality over time.")

    with col6:
        section("Attrition Rate by Travel Frequency")
        trav = attr_pct("BusinessTravel").sort_values("Left", ascending=True)
        fig = go.Figure(go.Bar(
            x=trav["Left"], y=trav["BusinessTravel"], orientation="h",
            marker=dict(
                color=[C_LEFT if v == trav["Left"].max() else C_ACCENT for v in trav["Left"]],
                line=dict(width=0)
            ),
            text=trav["Left"].map(lambda v: f"{v:.0f}%"),
            textposition="outside"
        ))
        lo = base_layout(height=270)
        lo["xaxis"]["range"]    = [0, trav["Left"].max() + 10]
        lo["xaxis"]["showgrid"] = False
        lo["yaxis"]["showgrid"] = False
        lo["plot_bgcolor"]      = "rgba(0,0,0,0)"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ins("Employees who travel frequently show the highest attrition. Frequent travel adds personal strain and makes competing offers more attractive.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Summary table
    section("Key Averages at a Glance")
    summary = (
        f.groupby("Status")[["MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
                              "DistanceFromHome", "YearsSinceLastPromotion", "Age"]]
        .mean().round(1)
        .rename(columns={
            "MonthlyIncome":           "Avg Monthly Pay",
            "JobSatisfaction":         "Job Satisfaction (1-4)",
            "WorkLifeBalance":         "Work-Life Balance (1-4)",
            "DistanceFromHome":        "Distance from Office (km)",
            "YearsSinceLastPromotion": "Yrs Since Last Promotion",
            "Age":                     "Average Age"
        })
    )
    st.dataframe(summary, use_container_width=True)
    ins("Employees who left score lower across every dimension. The biggest gaps are in pay and time since last promotion — both are directly actionable.")
