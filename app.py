
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
    st.subheader("Cumulative Return")
    df_filtered = df_filtered.copy()
    df_filtered["Return"] = df_filtered["PEP_Close"].pct_change()
    df_filtered["Cumulative Return"] = (1 + df_filtered["Return"]).cumprod()
    st.line_chart(df_filtered.set_index("Date")["Cumulative Return"])
    st.markdown("PEP shows steady long-term growth.")

elif section == "Volatility Analysis":
    st.subheader("30-Day Rolling Volatility")
    df_filtered = df_filtered.copy()
    df_filtered["Volatility"] = df_filtered["PEP_Close"].pct_change().rolling(30).std()
    st.line_chart(df_filtered.set_index("Date")["Volatility"])
    st.markdown("Volatility peaked during COVID, but remained moderate overall.")

elif section == "Forecasting Results":
    st.subheader("Forecasting Model Accuracy")
    st.bar_chart({
        "LSTM_Tuned": 0.5353,
        "XGBoost_Tuned": 0.5282,
        "ARIMA": 0.4640
    })
    st.markdown("LSTM is the most reliable model based on historical accuracy.")

elif section == "Financial Trends":
    st.subheader("Revenue, Net Income, and Liabilities")
    st.line_chart(df_fin_filtered.set_index("Year")[["Revenue", "NetIncome", "Liabilities"]])
    st.markdown("PepsiCo has shown strong growth in revenue and profitability.")

st.markdown("---")
st.caption("Built by PRIME INC • Powered by Streamlit")
