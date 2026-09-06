import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# CONFIG, PALETTE, CSS, helper functions here...

# DATA LOADING
df = load()  # or handle file upload fallback

# SIDEBAR
with st.sidebar:
    st.markdown("### HR Attrition Report")
    st.markdown("---")
    st.markdown("**Page**")
    page = st.radio("", ["Overview", "People & Roles", "Work Conditions"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Filters**")
    depts   = st.multiselect("Department", sorted(df["Department"].unique()),
                             default=list(df["Department"].unique()))
    genders = st.multiselect("Gender", sorted(df["Gender"].unique()),
                             default=list(df["Gender"].unique()))
    travel  = st.multiselect("Business Travel", sorted(df["BusinessTravel"].unique()),
                             default=list(df["BusinessTravel"].unique()))

# FILTERED DATA
f = df[df["Department"].isin(depts) &
       df["Gender"].isin(genders) &
       df["BusinessTravel"].isin(travel)].copy()

total  = len(f)
n_left = int(f["Attrition"].sum())
n_stay = total - n_left
rate   = round(n_left / total * 100, 1) if total > 0 else 0
# =====================================================================
# PAGE 1 — OVERVIEW
# =====================================================================
if page == "Overview":

    st.markdown('<p class="page-title">Attrition Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">A high-level summary of employee retention across the organisation.</p>', unsafe_allow_html=True)

    # KPI row (compact in 4 columns)
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

    # Retention Split
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # Attrition Rate by Department
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

    # Department Summary
    section("Department Summary")
    snap = (f.groupby("Department")
            .agg(Total=("Attrition","count"), Left=("Attrition","sum"))
            .assign(**{"Retention Rate (%)": lambda d: ((d["Total"]-d["Left"])/d["Total"]*100).round(1),
                       "Attrition Rate (%)": lambda d: (d["Left"]/d["Total"]*100).round(1)})
            .reset_index()
            .rename(columns={"Total": "Total Employees", "Left": "Employees Left"}))
    st.dataframe(snap, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Attrition by Gender
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

    # Age Group
    section("Attrition Rate by Age Group")
    age = attr_pct("AgeGroup")
    age["AgeGroup"] = age["AgeGroup"].astype(str)
    fig = go.Figure(go.Bar(
        x=age["AgeGroup"], y=age["Left"],
        marker_color=[C_LEFT if v == age["Left"].max() else C_ACCENT for v in age["Left"]],
        text=age["Left"].map(lambda v: f"{v:.0f}%"),
        textposition="outside"
    ))
    lo = base_layout(height=310)
    lo["yaxis"]["range"] = [0, age["Left"].max() + 12]
    lo["yaxis"]["title"] = "% who left"
    lo["xaxis"]["title"] = "Age Group"
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)
    top_age = age.loc[age["Left"].idxmax(), "AgeGroup"]
    ins(f"Employees aged {top_age} have the highest attrition.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Job Role
    section("Attrition Rate by Job Role")
    role = (f.groupby("JobRole")["Attrition"]
            .mean().mul(100).round(1).reset_index()
            .rename(columns={"Attrition": "Rate"})
            .sort_values("Rate", ascending=True))
    fig = go.Figure(go.Bar(
        x=role["Rate"], y=role["JobRole"], orientation="h",
        marker=dict(color=[C_LEFT if v == role["Rate"].max() else C_MID for v in role["Rate"]]),
        text=role["Rate"].map(lambda v: f"{v}%"),
        textposition="outside"
    ))
    lo = base_layout(height=310)
    lo["xaxis"]["range"]    = [0, role["Rate"].max() + 10]
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)
    top_role = role.iloc[-1]
    ins(f"{top_role['JobRole']} has the highest attrition at {top_role['Rate']}%.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tenure Group
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
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)
    top_ten = ten.loc[ten["Left"].idxmax(), "TenureGroup"]
    ins(f"Employees in their {top_ten} are most likely to leave.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Promotion Gap
    section("Avg Years Since Last Promotion")
    promo = f.groupby("Status")["YearsSinceLastPromotion"].mean().round(1)
