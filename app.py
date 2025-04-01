
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
    "Financial Trends",
    "Regression Analysis"
])
year_range = st.sidebar.slider("Select Year Range", 2010, 2024, (2015, 2024))
df_filtered = df_stock[(df_stock["Year"] >= year_range[0]) & (df_stock["Year"] <= year_range[1])]
df_fin_filtered = df_financials[(df_financials["Year"] >= year_range[0]) & (df_financials["Year"] <= year_range[1])]

st.title("PepsiCo Investment Insights")

if section == "Stock Return Overview":
    st.subheader("Cumulative Returns")

    options = st.sidebar.multiselect("Select which stocks to display", 
        ["PepsiCo (PEP)", "Coca-Cola (KO)", "S&P 500"], 
        default=["PepsiCo (PEP)", "Coca-Cola (KO)", "S&P 500"]
    )

    name_map = {
        "PepsiCo (PEP)": "PEP_CumReturn",
        "Coca-Cola (KO)": "KO_CumReturn",
        "S&P 500": "GSPC_CumReturn"
    }

    df_filtered = df_filtered.copy()
    df_filtered.dropna(subset=["PEP_Return", "KO_Return", "GSPC_Return"], inplace=True)
    df_filtered["PEP_CumReturn"] = (1 + df_filtered["PEP_Return"]).cumprod()
    df_filtered["KO_CumReturn"] = (1 + df_filtered["KO_Return"]).cumprod()
    df_filtered["GSPC_CumReturn"] = (1 + df_filtered["GSPC_Return"]).cumprod()

    selected_columns = [name_map[o] for o in options]
    chart_df = df_filtered[selected_columns].rename(columns={v: k for k, v in name_map.items()})
    chart_df.index = pd.to_datetime(df_filtered["Date"]) if "Date" in df_filtered else df_filtered.index
    st.line_chart(chart_df)

elif section == "Volatility Analysis":
    st.subheader("30-Day Rolling Volatility")

    options = st.sidebar.multiselect("Select which volatilities to display", 
        ["PepsiCo (PEP)", "Coca-Cola (KO)", "S&P 500"], 
        default=["PepsiCo (PEP)", "Coca-Cola (KO)", "S&P 500"]
    )

    name_map = {
        "PepsiCo (PEP)": "PEP_RollingVol",
        "Coca-Cola (KO)": "KO_RollingVol",
        "S&P 500": "GSPC_RollingVol"
    }

    df_filtered = df_filtered.copy()
    df_filtered.dropna(subset=list(name_map.values()), inplace=True)

    selected_columns = [name_map[o] for o in options]
    vol_df = df_filtered[selected_columns].rename(columns={v: k for k, v in name_map.items()})
    vol_df.index = pd.to_datetime(df_filtered["Date"]) if "Date" in df_filtered else df_filtered.index
    st.line_chart(vol_df)

elif section == "Forecasting Results":
    st.subheader("Forecasting Model Accuracy from MLflow Logs")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name("PEP_Stock_Trend_Forecasting")

    runs = client.search_runs(experiment.experiment_id)

    all_scores = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        accuracy = run.data.metrics.get("accuracy", None)
        if accuracy is not None:
            all_scores[name] = accuracy

    if all_scores:
        model_names = list(all_scores.keys())
        selected_models = st.sidebar.multiselect("Select models to display", model_names, default=model_names)
        filtered_scores = {k: all_scores[k] for k in selected_models if k in all_scores}
        st.bar_chart(filtered_scores)

        if filtered_scores:
            best_model = max(filtered_scores, key=filtered_scores.get)
            st.markdown(f"**Best Performing Model:** {best_model} ({filtered_scores[best_model]:.2%})")
    else:
        st.warning("No model accuracies found in MLflow logs.")

elif section == "Financial Trends":
    st.subheader("Financial Trends Over Time")
    available_metrics = [col for col in df_fin_filtered.columns if col != "Year"]
    selected_metrics = st.sidebar.multiselect(
        "Select metrics to display", available_metrics, default=available_metrics[:3]
    )

    if selected_metrics:
        st.line_chart(df_fin_filtered.set_index("Year")[selected_metrics])
        st.markdown("PepsiCo's financial performance over time.")
    else:
        st.warning("Please select at least one financial metric to display.")

elif section == "Regression Analysis":
    st.subheader("Market Return vs Stock Return (OLS Regression)")

    import statsmodels.api as sm

    # Prepare data
    X = df_stock["GSPC_Return"]
    X = sm.add_constant(X)

    y_pep = df_stock["PEP_Return"]
    y_ko = df_stock["KO_Return"]

    model_pep = sm.OLS(y_pep, X, missing="drop").fit()
    model_ko = sm.OLS(y_ko, X, missing="drop").fit()

    # Show summaries
    st.text("PEP Regression Summary:")
    st.text(model_pep.summary().as_text())

    st.text("KO Regression Summary:")
    st.text(model_ko.summary().as_text())

    # Simple, clean interpretation
    st.markdown("### Interpretation")
    st.markdown("R-squared shows how well a stock’s return is explained by the S&P 500 (market).")
    st.markdown(f"**PEP R²**: {model_pep.rsquared:.3f} | **KO R²**: {model_ko.rsquared:.3f}")

    # Logic to handle tie, higher PEP, or higher KO
    alignment_message = (
        "Both stocks are equally aligned with market movement."
        if model_pep.rsquared == model_ko.rsquared else
        "PepsiCo (PEP) is more aligned with market movement."
        if model_pep.rsquared > model_ko.rsquared else
        "Coca-Cola (KO) is more aligned with market movement."
    )
    st.markdown(alignment_message)

    st.markdown("Keep in mind that lower correlation could offer more diversification benefits, depending on your investment strategy.")

st.markdown("---")
st.caption("Built by PRIME INC • Powered by Streamlit")
