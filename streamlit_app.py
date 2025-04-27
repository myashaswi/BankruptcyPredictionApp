# ========================
# Bankruptcy Prediction App
# Final Streamlit Code
# ========================

import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set page configuration
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")

# Define page options
page = st.sidebar.radio(
    "Navigation - Go to:",
    [
        "1 About this App",
        "2 Bankruptcy Prediction Based on Ticker",
        "3 Model Training Code",
        "5 Full Streamlit App Code",
        "6 Manually Enter Data"
    ]
)

# ----------- PAGE 1: About this App ----------- #
if page.startswith("1"):
    st.title("Altman Z-Score Inspired Industry-Specific Model")

    st.subheader("What, How, Why?")
    st.markdown(
        f"""
        <h3 style="color:#c95c5d;">What?</h3>
        This app predicts bankruptcy risk using a customized logistic regression model with <b>industry-specific intercepts (alphas)</b> and <b>key financial ratios (betas)</b>.

        <h3 style="color:#c95c5d;">Why?</h3>
        Early prediction of bankruptcy risk helps prioritize audits, deeper analysis, and proactive decision-making.

        <h3 style="color:#c95c5d;">How?</h3>
        We trained a model using:
        <ul>
            <li>Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.</li>
            <li>Industry dummy variables for 10 industries</li>
            <li>Logistic regression combining both to predict 5-year bankruptcy likelihood</li>
        </ul>

        <br><br>
        <b>Expanded form:</b><br>
        <div style="font-size:20px;"> 
        z = Σ(αᵢ × industryᵢ) + Σ(βⱼ × ratioⱼ)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("**Model trained as of April 25, 2025.**")
    st.markdown("---")
    st.markdown("🔗 [GitHub Repository Link Here](github.com/myashaswi/BankruptcyPredictionApp/)")

# ----------- PAGE 2: Bankruptcy Prediction Based on Ticker ----------- #
elif page.startswith("2"):
    st.title("Bankruptcy Risk Prediction")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):").upper()

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            bs = stock.balance_sheet
            fs = stock.financials
            info = stock.info

            required_fields = [
                'Total Assets', 'Total Current Assets', 'Gross PPE', 'Accounts Receivable',
                'Total Revenue', 'Operating Cash Flow', 'Net Income From Continuing Operations', 'Total Debt'
            ]

            if not all(field in bs.index for field in required_fields[:4]) or not all(field in fs.index for field in required_fields[4:]):
                raise ValueError("Missing critical financial fields. Cannot compute ratios.")

            # Ratios
            working_capital_ratio = ((bs.loc['Total Current Assets'] - (bs.loc['Total Assets'] - bs.loc['Total Current Assets'])) / bs.loc['Total Assets']).iloc[0]
            roa = (fs.loc['Net Income From Continuing Operations'] / bs.loc['Total Assets']).iloc[0]
            ebit_to_assets = (fs.loc['Operating Cash Flow'] / bs.loc['Total Assets']).iloc[0]
            debt_to_equity = (bs.loc['Total Debt'] / (bs.loc['Total Assets'] - bs.loc['Total Debt'])).iloc[0]
            interest_coverage = (fs.loc['Operating Cash Flow'] / (fs.loc['Operating Cash Flow'] - fs.loc['Net Income From Continuing Operations'])).iloc[0]
            ocf_to_debt = (fs.loc['Operating Cash Flow'] / bs.loc['Total Debt']).iloc[0]
            receivables_turnover = (fs.loc['Total Revenue'] / bs.loc['Accounts Receivable']).iloc[0]
            payables_turnover_days = (bs.loc['Total Assets'] / bs.loc['Gross PPE']).iloc[0]

            input_ratios = pd.DataFrame([[
                working_capital_ratio, roa, ebit_to_assets, debt_to_equity,
                interest_coverage, ocf_to_debt, receivables_turnover, payables_turnover_days
            ]], columns=[
                'working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
                'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days'
            ])

            st.subheader("Fetched Financial Data:")
            st.write(f"**Company:** {info.get('longName', 'Unknown')}")
            st.write(f"**Industry:** {info.get('industry', 'Unknown')}")

            st.subheader("Input Ratios:")
            st.dataframe(input_ratios)

            # Scaling and prediction
            input_scaled = scaler.transform(input_ratios)
            prediction = model.predict_proba(input_scaled)[0][1]

            st.subheader("Prediction:")
            color = "#c95c5d" if prediction >= 0.5 else "green"
            st.markdown(
                f"<h2 style='color:{color};'>{'High' if prediction >= 0.5 else 'Low'} Bankruptcy Risk ({prediction*100:.2f}%)</h2>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing data for {ticker}. Details: {str(e)}")

# ----------- PAGE 3: Model Training Code ----------- #
elif page.startswith("3"):
    st.title("Model Training Code (Model.ipynb)")

    with open("Model.ipynb", "r") as f:
        content = f.read()

    st.code(content, language="json")  # because .ipynb is JSON structured

# ----------- PAGE 5: Full Streamlit App Code ----------- #
elif page.startswith("5"):
    st.title("Full Streamlit App Code")

    try:
        with open("streamlit_app.py", "r") as f:
            code = f.read()
        st.code(code, language="python")
    except Exception as e:
        st.error(f"Unable to load code: {e}")

# ----------- PAGE 6: Manually Enter Data ----------- #
elif page.startswith("6"):
    st.title("Manual Entry of Financial Ratios")

    st.markdown("Fill the below values to manually predict bankruptcy:")

    col1, col2 = st.columns(2)

    with col1:
        working_capital_ratio = st.number_input("Working Capital Ratio")
        roa = st.number_input("Return on Assets (ROA)")
        ebit_to_assets = st.number_input("EBIT to Assets")
        debt_to_equity = st.number_input("Debt to Equity Ratio")

    with col2:
        interest_coverage = st.number_input("Interest Coverage")
        ocf_to_debt = st.number_input("Operating Cash Flow to Debt")
        receivables_turnover = st.number_input("Receivables Turnover")
        payables_turnover_days = st.number_input("Payables Turnover Days")

    if st.button("Predict Bankruptcy Risk"):
        try:
            manual_input = pd.DataFrame([[
                working_capital_ratio, roa, ebit_to_assets, debt_to_equity,
                interest_coverage, ocf_to_debt, receivables_turnover, payables_turnover_days
            ]], columns=[
                'working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
                'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days'
            ])

            scaled_manual = scaler.transform(manual_input)
            prediction_manual = model.predict_proba(scaled_manual)[0][1]

            color = "#c95c5d" if prediction_manual >= 0.5 else "green"
            st.markdown(
                f"<h2 style='color:{color};'>{'High' if prediction_manual >= 0.5 else 'Low'} Bankruptcy Risk ({prediction_manual*100:.2f}%)</h2>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error in prediction: {e}")
