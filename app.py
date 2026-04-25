import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# ──────────────────────────────────────────────
# CUSTOM STYLE  (light, clean)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .insight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 15px;
        color: #1e3a5f;
    }
    .kpi-label {font-size:13px; color:#6b7280; margin-bottom:2px;}
    .kpi-value {font-size:28px; font-weight:600; color:#111827;}
    .kpi-note  {font-size:12px; color:#6b7280; margin-top:2px;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_hr_data.csv")
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    bins   = [18, 25, 35, 45, 55, 60]
    labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df

try:
    df = load_data()
except FileNotFoundError:
    uploaded = st.sidebar.file_uploader("Upload HR CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if df["Attrition"].dtype == object:
            df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
        bins   = [18, 25, 35, 45, 55, 60]
        labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    else:
        st.warning("Please upload your HR dataset (CSV) using the sidebar.")
        st.stop()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("Filters")
    st.caption("Use the filters below to explore specific groups of employees.")

    department = st.multiselect(
        "Department",
        df["Department"].unique(),
        default=list(df["Department"].unique()),
        help="Select one or more departments to focus on."
    )
    overtime = st.multiselect(
        "Overtime",
        df["OverTime"].unique(),
        default=list(df["OverTime"].unique()),
        help="Filter by whether employees work overtime."
    )

    st.divider()
    page = st.radio(
        "Go to",
        ["Overview", "Deep Dive", "Predict"],
        format_func=lambda x: {
            "Overview": "📊  Overview",
            "Deep Dive": "🔍  Deep Dive",
            "Predict":  "🎯  Predict"
        }[x]
    )

filtered = df[
    df["Department"].isin(department) &
    df["OverTime"].isin(overtime)
]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
LABEL_MAP = {0: "Stayed", 1: "Left"}
COLOR_MAP  = {"Stayed": "#22c55e", "Left": "#ef4444"}

def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def attrition_rate(data):
    return round(data["Attrition"].mean() * 100, 1)

# ======================================================
# PAGE 1 — OVERVIEW
# ======================================================
if page == "Overview":

    st.title("Employee Attrition Dashboard")
    st.caption("A simple overview of who is leaving and why — no technical background needed.")

    total  = len(filtered)
    left   = int(filtered["Attrition"].sum())
    stayed = total - left
    rate   = attrition_rate(filtered)

    # KPI cards with colour-coded attrition
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Employees Still Here", f"{stayed:,}")

    if rate >= 20:
        col3.metric("Attrition Rate", f"{rate}%", delta="High – needs attention", delta_color="inverse")
    elif rate >= 10:
        col3.metric("Attrition Rate", f"{rate}%", delta="Moderate – monitor closely", delta_color="off")
    else:
        col3.metric("Attrition Rate", f"{rate}%", delta="Low – good retention", delta_color="normal")

    st.divider()

    # Quick snapshot – readable summary table
    st.subheader("Quick snapshot by department")
    snap = (
        filtered.groupby("Department")
        .agg(
            Total=("Attrition", "count"),
            Left=("Attrition", "sum")
        )
        .assign(Attrition_Rate=lambda d: (d["Left"] / d["Total"] * 100).round(1))
        .rename(columns={"Attrition_Rate": "Attrition Rate (%)"})
        .reset_index()
    )
    st.dataframe(snap, use_container_width=True, hide_index=True)

    insight(
        f"Out of {total:,} employees, {left:,} have left — that's {rate}% attrition. "
        "Use the filters in the sidebar to drill into specific departments or overtime groups."
    )

# ======================================================
# PAGE 2 — DEEP DIVE
# ======================================================
elif page == "Deep Dive":

    st.title("Why Are Employees Leaving?")
    st.caption("Each chart below shows a different factor. Insights are written in plain English below every chart.")

    # — Attrition split
    st.subheader("Overall: Who stayed and who left?")
    counts = filtered["Attrition"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Status"] = counts["Status"].map(LABEL_MAP)
    fig = px.pie(counts, names="Status", values="Count", hole=0.55,
                 color="Status", color_discrete_map=COLOR_MAP)
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)
    rate = attrition_rate(filtered)
    insight(f"{rate}% of employees have left. The rest are still with the company.")

    st.divider()

    # — Department
    st.subheader("Which department has the most attrition?")
    dept = (
        filtered.groupby("Department")["Attrition"]
        .mean()
        .mul(100).round(1)
        .reset_index()
        .rename(columns={"Attrition": "Attrition Rate (%)"})
        .sort_values("Attrition Rate (%)", ascending=False)
    )
    fig = px.bar(dept, x="Department", y="Attrition Rate (%)",
                 color="Department", text="Attrition Rate (%)",
                 labels={"Attrition Rate (%)": "% who left"})
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, dept["Attrition Rate (%)"].max() + 10])
    st.plotly_chart(fig, use_container_width=True)
    top = dept.iloc[0]
    insight(
        f"The {top['Department']} department has the highest attrition at {top['Attrition Rate (%)']:.1f}%. "
        "This is where retention efforts should be focused first."
    )

    st.divider()

    # — Salary
    st.subheader("Does pay affect whether someone leaves?")
    salary_df = filtered.copy()
    salary_df["Status"] = salary_df["Attrition"].map(LABEL_MAP)
    fig = px.box(salary_df, x="Status", y="MonthlyIncome",
                 color="Status", color_discrete_map=COLOR_MAP,
                 labels={"MonthlyIncome": "Monthly Income (₹)", "Status": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    avg = filtered.groupby("Attrition")["MonthlyIncome"].mean()
    insight(
        f"Employees who left earned an average of ₹{int(avg[1]):,}/month, "
        f"compared to ₹{int(avg[0]):,}/month for those who stayed. "
        "Lower pay is clearly linked to higher turnover."
    )

    st.divider()

    # — Overtime
    st.subheader("Does working overtime push people to leave?")
    ot = (
        pd.crosstab(filtered["OverTime"], filtered["Attrition"], normalize="index") * 100
    ).reset_index()
    ot.columns = ["OverTime", "Stayed (%)", "Left (%)"]
    fig = px.bar(ot, x="OverTime", y="Left (%)", color="OverTime",
                 text="Left (%)", labels={"Left (%)": "% who left", "OverTime": "Works Overtime?"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, ot["Left (%)"].max() + 15])
    st.plotly_chart(fig, use_container_width=True)
    if "Yes" in ot["OverTime"].values:
        ot_rate = ot.loc[ot["OverTime"] == "Yes", "Left (%)"].values[0]
        insight(
            f"{ot_rate:.1f}% of overtime workers left the company — "
            "almost 3× more than those who don't work overtime. "
            "Excessive workload is a major attrition driver."
        )

    st.divider()

    # — Age group
    st.subheader("Which age group leaves the most?")
    age_df = filtered.copy()
    age_df["Status"] = age_df["Attrition"].map(LABEL_MAP)
    fig = px.histogram(age_df, x="AgeGroup", color="Status",
                       barmode="group", color_discrete_map=COLOR_MAP,
                       labels={"AgeGroup": "Age Group", "count": "Number of Employees", "Status": ""})
    st.plotly_chart(fig, use_container_width=True)
    age_left = filtered.groupby("AgeGroup")["Attrition"].sum()
    top_age  = age_left.idxmax()
    insight(
        f"Employees aged {top_age} leave the most. "
        "This age group often seeks career growth or higher salaries elsewhere."
    )

    st.divider()

    # — Job satisfaction (plain language labels)
    st.subheader("Does job satisfaction matter?")
    sat_df = filtered.copy()
    sat_df["Status"] = sat_df["Attrition"].map(LABEL_MAP)
    sat_df["Satisfaction Label"] = sat_df["JobSatisfaction"].map(
        {1: "Very Low", 2: "Low", 3: "High", 4: "Very High"}
    )
    fig = px.histogram(sat_df, x="Satisfaction Label",
                       color="Status", barmode="group",
                       color_discrete_map=COLOR_MAP,
                       category_orders={"Satisfaction Label": ["Very Low", "Low", "High", "Very High"]},
                       labels={"count": "Number of Employees", "Status": ""})
    st.plotly_chart(fig, use_container_width=True)
    sat_avg = filtered.groupby("Attrition")["JobSatisfaction"].mean()
    insight(
        f"Employees who left rated their job satisfaction {sat_avg[1]:.1f}/4 on average, "
        f"vs {sat_avg[0]:.1f}/4 for those who stayed. "
        "Improving satisfaction could significantly reduce turnover."
    )

    st.divider()

    # — Summary table
    st.subheader("Key averages at a glance")
    factors = (
        filtered.groupby("Attrition")[
            ["MonthlyIncome", "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome", "Age"]
        ]
        .mean()
        .round(1)
        .rename(index=LABEL_MAP)
        .rename(columns={
            "MonthlyIncome": "Avg Monthly Pay (₹)",
            "JobSatisfaction": "Job Satisfaction (1-4)",
            "WorkLifeBalance": "Work-Life Balance (1-4)",
            "DistanceFromHome": "Avg Distance from Home (km)",
            "Age": "Average Age"
        })
    )
    st.dataframe(factors, use_container_width=True)
    insight(
        "Employees who left consistently scored lower on pay, satisfaction, and work-life balance, "
        "and lived farther from the office."
    )

# ======================================================
# PAGE 3 — PREDICT
# ======================================================
elif page == "Predict":

    st.title("Will This Employee Leave?")
    st.caption(
        "Fill in a few details about an employee and we'll estimate their risk of leaving. "
        "This uses patterns learned from your existing HR data."
    )

    # Train model (cached)
    @st.cache_resource
    def train_model(data):
        feats = ["Age", "MonthlyIncome", "DistanceFromHome", "JobSatisfaction", "WorkLifeBalance"]
        X = data[feats]
        y = data["Attrition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        return clf, round(acc * 100, 1)

    model, acc = train_model(filtered)

    st.caption(f"Model accuracy on test data: **{acc}%** — trained on {len(filtered):,} employees.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age      = st.slider("Employee Age", 18, 60, 30)
        salary   = st.number_input("Monthly Income (₹)", min_value=1000, max_value=20000,
                                   value=5000, step=500)
        distance = st.slider("Distance from Home (km)", 1, 30, 10)

    with col2:
        st.markdown("**Job Satisfaction** — How happy is the employee with their role?")
        job_sat_label = st.select_slider(
            "Job Satisfaction",
            options=["Very Low (1)", "Low (2)", "High (3)", "Very High (4)"],
            value="Low (2)",
            label_visibility="collapsed"
        )
        job_sat = {"Very Low (1)": 1, "Low (2)": 2, "High (3)": 3, "Very High (4)": 4}[job_sat_label]

        st.markdown("**Work-Life Balance** — How well does work fit into their personal life?")
        wlb_label = st.select_slider(
            "Work-Life Balance",
            options=["Very Poor (1)", "Poor (2)", "Good (3)", "Excellent (4)"],
            value="Poor (2)",
            label_visibility="collapsed"
        )
        wlb = {"Very Poor (1)": 1, "Poor (2)": 2, "Good (3)": 3, "Excellent (4)": 4}[wlb_label]

    st.divider()

    if st.button("Check attrition risk", type="primary"):
        input_df = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [salary],
            "DistanceFromHome": [distance],
            "JobSatisfaction": [job_sat],
            "WorkLifeBalance": [wlb]
        })
        prob = model.predict_proba(input_df)[0][1] * 100

        if prob >= 60:
            st.error(f"⚠️ High risk of leaving ({prob:.0f}% probability). Consider a retention conversation.")
        elif prob >= 35:
            st.warning(f"🟡 Moderate risk ({prob:.0f}% probability). Worth keeping an eye on this employee.")
        else:
            st.success(f"✅ Low risk of leaving ({prob:.0f}% probability). This employee looks stable.")

        st.caption(
            "This is an estimate based on historical patterns — not a guarantee. "
            "Use it as a conversation starter, not a final decision."
        )