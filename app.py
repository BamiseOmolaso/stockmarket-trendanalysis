
# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_stock = pd.read_csv("pep_stock_data.csv")
df_financials = pd.read_csv("pep_financials.csv")

# Parse dates and years
df_stock["Date"] = pd.to_datetime(df_stock["Date"])
df_stock["Year"] = df_stock["Date"].dt.year

# Sidebar controls
st.sidebar.title("PepsiCo Investment Dashboard")
section = st.sidebar.selectbox("Select Analysis Section", [
    "Stock Return Overview", 
    "Volatility Analysis", 
    "Forecasting Results", 
    "Financial Trends"
])
year_range = st.sidebar.slider("Select Year Range", 2010, 2024, (2015, 2024))
df_filtered = df_stock[(df_stock["Year"] >= year_range[0]) & (df_stock["Year"] <= year_range[1])]
df_fin_filtered = df_financials[(df_financials["Year"] >= year_range[0]) & (df_financials["Year"] <= year_range[1])]

st.title("PepsiCo Investment Insights")

if section == "Stock Return Overview":
    st.subheader("Cumulative Returns: PEP vs KO vs S&P 500")

    df_filtered = df_filtered.copy()
    df_filtered.dropna(subset=["PEP_Return", "KO_Return", "GSPC_Return"], inplace=True)

    # Calculate cumulative returns
    df_filtered["PEP_CumReturn"] = (1 + df_filtered["PEP_Return"]).cumprod()
    df_filtered["KO_CumReturn"] = (1 + df_filtered["KO_Return"]).cumprod()
    df_filtered["GSPC_CumReturn"] = (1 + df_filtered["GSPC_Return"]).cumprod()

    # Rename columns for nicer labels
    chart_df = df_filtered[["PEP_CumReturn", "KO_CumReturn", "GSPC_CumReturn"]].rename(columns={
        "PEP_CumReturn": "PepsiCo (PEP)",
        "KO_CumReturn": "Coca-Cola (KO)",
        "GSPC_CumReturn": "S&P 500"
    })

    chart_df.index = pd.to_datetime(df_filtered["Date"]) if "Date" in df_filtered else df_filtered.index
    st.line_chart(chart_df)

    st.markdown("""
    - **PepsiCo (PEP)**: Primary investment target
    - **Coca-Cola (KO)**: Competitor
    - **S&P 500**: Market benchmark
    """)

elif section == "Volatility Analysis":
    st.subheader("30-Day Rolling Volatility: PEP vs KO vs S&P 500")
    df_filtered = df_filtered.copy()
    df_filtered.dropna(subset=["PEP_RollingVol", "KO_RollingVol", "GSPC_RollingVol"], inplace=True)

    vol_df = df_filtered[["PEP_RollingVol", "KO_RollingVol", "GSPC_RollingVol"]].rename(columns={
        "PEP_RollingVol": "PepsiCo (PEP)",
        "KO_RollingVol": "Coca-Cola (KO)",
        "GSPC_RollingVol": "S&P 500"
    })
    vol_df.index = pd.to_datetime(df_filtered["Date"]) if "Date" in df_filtered else df_filtered.index
    st.line_chart(vol_df)

    st.markdown("""
    - Rolling 30-day standard deviation of daily returns.
    - Helps assess recent market risk trends.
    """)

elif section == "Forecasting Results":
    st.subheader("Forecasting Model Accuracy from MLflow Logs")

    import mlflow
    from mlflow.tracking import MlflowClient

    # Load MLflow logs from your local experiment
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name("PEP_Stock_Trend_Forecasting")

    runs = client.search_runs(experiment.experiment_id)

    model_scores = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        accuracy = run.data.metrics.get("accuracy", None)
        if accuracy:
            model_scores[name] = accuracy

    if model_scores:
        st.bar_chart(model_scores)
        best_model = max(model_scores, key=model_scores.get)
        st.markdown(f"**Best Performing Model:** {best_model} ({model_scores[best_model]:.2%})")
    else:
        st.warning("No model accuracies found in MLflow logs.")

elif section == "Financial Trends":
    st.subheader("Revenue, Net Income, Gross Profit, Assets and Liabilities")
    st.line_chart(df_fin_filtered.set_index("Year")[[
        "Revenue", "Net_Income", "Gross_Profit", "Total_Assets", "Total_Liabilities"
    ]])
    st.markdown("PepsiCo has shown strong growth in revenue and profitability.")

st.markdown("---")
st.caption("Built by PRIME INC • Powered by Streamlit")
