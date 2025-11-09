import streamlit as st
import pandas as pd
import plotly.express as px
from src.churn.utils import load_data, clean_data, train_models, kpis, feature_importance_from_rf

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("Customer Churn Analysis Dashboard")

with st.sidebar:
    st.header("Controls")
    model_choice = st.selectbox("Model", ["random_forest", "log_reg"], index=0)
    st.caption("Dataset: IBM Telco Customer Churn (auto-downloaded)")

@st.cache_data(show_spinner=False)
def get_data():
    df_raw = load_data()
    df = clean_data(df_raw)
    return df

@st.cache_resource(show_spinner=True)
def get_models():
    df = load_data()
    results = train_models(df)
    return results

# Load data and KPIs
with st.spinner("Loading data..."):
    df = get_data()
    metrics = kpis(df)

# KPI cards
kpi_cols = st.columns(5)
kpi_cols[0].metric("Churn Rate", f"{metrics['churn_rate']:.1f}%")
kpi_cols[1].metric("Retention Rate", f"{metrics['retention_rate']:.1f}%")
kpi_cols[2].metric("Avg Tenure", f"{metrics['avg_tenure']:.1f} mo")
kpi_cols[3].metric("High-Risk Share", f"{metrics['high_risk_share']:.1f}%")
kpi_cols[4].metric("CLV (avg)", f"${metrics['clv_estimate']:.0f}")

# Churn by key dimensions
left, right = st.columns(2)

with left:
    fig1 = px.bar(
        df.groupby("Contract")["Churn"].mean().mul(100).reset_index(),
        x="Contract", y="Churn", title="Churn Rate by Contract Type", text="Churn"
    )
    fig1.update_traces(texttemplate='%{text:.1f}%')
    st.plotly_chart(fig1, use_container_width=True)

with right:
    if "PaymentMethod" in df.columns:
        fig2 = px.bar(
            df.groupby("PaymentMethod")["Churn"].mean().mul(100).reset_index(),
            x="PaymentMethod", y="Churn", title="Churn Rate by Payment Method", text="Churn"
        )
        fig2.update_traces(texttemplate='%{text:.1f}%')
        st.plotly_chart(fig2, use_container_width=True)

# Tenure vs Churn
fig3 = px.histogram(
    df, x="tenure", color=df["Churn"].map({1: "Churned", 0: "Stayed"}), barmode="overlay",
    title="Tenure Distribution by Churn"
)
fig3.update_traces(opacity=0.6)
st.plotly_chart(fig3, use_container_width=True)

# Modeling section
st.subheader("Predictive Modeling")
with st.spinner("Training models (cached)..."):
    results = get_models()

sel = results[model_choice]
st.write(f"AUC: {sel['auc']:.3f}")

# Feature importances for RF
if model_choice == "random_forest":
    try:
        rf_imp = feature_importance_from_rf(sel["model"], df.drop(columns=["Churn"]).columns)
        fig_imp = px.bar(rf_imp.head(20), x="importance", y="feature", orientation="h", title="Top Features")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.info("Feature importance not available.")

# Data preview
with st.expander("Preview Data"):
    st.dataframe(df.head(50))
