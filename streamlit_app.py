import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import joblib
import numpy as np
import plotly.graph_objects as go

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = joblib.load("scaler.pkl")

# Define financial ratios needed
required_fields = {
    "Total Current Assets": "current_assets",
    "Total Current Liabilities": "current_liabilities",
    "Total Assets": "total_assets",
    "Total Debt": "total_debt",
    "EBIT": "ebit",
    "Net Income": "net_income",
    "Operating Cash Flow": "ocf",
    "Accounts Receivable": "accounts_receivable",
    "Gross PPE": "gross_ppe",
    "Total Revenue": "revenue"
}

# Sidebar Navigation
page = st.sidebar.radio("Navigation - Go to:", [
    "1 About this App",
    "2 Bankruptcy Prediction Based on Ticker",
    "3 Model Training Code",
    "4 Full Streamlit App Code",
    "5 Manually Enter Data"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ in #c95c5d")

# Set Red  Title
Red = "#c95c5d"

if page.startswith("1"):
    st.title("Altman Z-Score Inspired Industry-Specific Model")
    st.markdown(f"<h2 style='color:{prussian_red};'>What, How, Why?</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='color:{prussian_red};'>What?</h3>", unsafe_allow_html=True)
    st.write("This app predicts bankruptcy risk using a customized logistic regression model, designed with **industry-specific intercepts (alphas)** and **key financial ratios (betas)**.")

    st.markdown(f"<h3 style='color:{prussian_red};'>Why?</h3>", unsafe_allow_html=True)
    st.write("Predicting bankruptcy risk early can help prioritize audits, deeper analysis, and proactive financial decisions.")

    st.markdown(f"<h3 style='color:{prussian_red};'>How?</h3>", unsafe_allow_html=True)
    st.write("""
    We built a model using:
    - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
    - Industry dummy variables for 10 industries
    - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.markdown("Expanded Form of Model:")
    st.latex(r"""
    z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j
    """)
    st.markdown("Model trained as of **April 25, 2025**.")

    st.markdown("---")
    st.markdown("🔗 [GitHub Repository for this App](https://github.com/myashaswi/BankruptcyPredictionApp/)")

elif page.startswith("2"):
    st.title("Bankruptcy Risk Prediction")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):").upper()

    if ticker:
        try:
            stock = yf.Ticker(ticker)

            # Fetch required financials
            fin = stock.financials
            bs = stock.balance_sheet
            cf = stock.cashflow
            info = stock.info

            data = {}

            # Extracting required fields
            try:
                data["current_assets"] = bs.loc["Total Current Assets"].iloc[0]
                data["current_liabilities"] = bs.loc["Total Current Liabilities"].iloc[0]
                data["total_assets"] = bs.loc["Total Assets"].iloc[0]
                data["total_debt"] = bs.loc["Total Debt"].iloc[0]
                data["ebit"] = fin.loc["EBIT"].iloc[0]
                data["net_income"] = fin.loc["Net Income"].iloc[0]
                data["ocf"] = cf.loc["Total Cash From Operating Activities"].iloc[0]
                data["accounts_receivable"] = bs.loc["Accounts Receivable"].iloc[0]
                data["gross_ppe"] = bs.loc["Gross Property Plant Equipment"].iloc[0]
                data["revenue"] = fin.loc["Total Revenue"].iloc[0]
            except Exception as e:
                st.error(f"Error fetching data for {ticker}. Details: {e}")
                st.stop()

            # Calculate Ratios
            try:
                ratios = {}
                ratios['working_capital_ratio'] = (data["current_assets"] - data["current_liabilities"]) / data["total_assets"]
                ratios['roa'] = data["net_income"] / data["total_assets"]
                ratios['ebit_to_assets'] = data["ebit"] / data["total_assets"]
                ratios['debt_to_equity'] = data["total_debt"] / (data["total_assets"] - data["total_debt"])
                ratios['interest_coverage'] = data["ebit"] / (data["total_debt"] * 0.05)  # Approx 5% cost
                ratios['ocf_to_debt'] = data["ocf"] / data["total_debt"]
                ratios['receivables_turnover'] = data["revenue"] / data["accounts_receivable"]
                ratios['payables_turnover_days'] = 365 / (data["revenue"] / (data["total_assets"] - data["gross_ppe"]))

                # Prepare dataframe
                input_df = pd.DataFrame([ratios])

                st.subheader("Fetched Financial Data:")
                st.write(f"**Company:** {info.get('longName', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.subheader("Input Ratios:")
                st.dataframe(input_df)

                # Scaling and Prediction
                scaled = scaler.transform(input_df)
                pred = model.predict_proba(scaled)[0][1]

                pred_percent = pred * 100

                # Plot gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_percent,
                    title={'text': "Bankruptcy Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': prussian_red},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing financial ratios for {ticker}. Details: {e}")

# ----------- PAGE 3: Model Training Code (Model.ipynb) ----------- #
elif page.startswith("3"):
    st.title("Model Training Code (From Model.ipynb)")

    try:
        with open("Model.ipynb", "r") as f:
            content = f.read()
        st.code(content, language="json")  # Display raw .ipynb file
    except Exception as e:
        st.error(f"Could not load Model.ipynb. Error: {e}")

# ----------- PAGE 4: Full Streamlit App Code (This file) ----------- #
elif page.startswith("4"):
    st.title("Full Streamlit App Code")

    try:
        with open("streamlit_app.py", "r") as f:
            code = f.read()
        st.code(code, language="python")
    except Exception as e:
        st.error(f"Could not load streamlit_app.py. Error: {e}")

# ----------- PAGE 5: Manually Enter Data ----------- #
elif page.startswith("5"):
    st.title("Manually Enter Financial Ratios")

    st.markdown(
        f"<h3 style='color:{prussian_red};'>Enter the following 8 ratios manually:</h3>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        working_capital_ratio = st.number_input("Working Capital Ratio", step=0.01)
        roa = st.number_input("Return on Assets (ROA)", step=0.01)
        ebit_to_assets = st.number_input("EBIT to Assets", step=0.01)
        debt_to_equity = st.number_input("Debt to Equity Ratio", step=0.01)

    with col2:
        interest_coverage = st.number_input("Interest Coverage", step=0.01)
        ocf_to_debt = st.number_input("Operating Cash Flow to Debt", step=0.01)
        receivables_turnover = st.number_input("Receivables Turnover", step=0.01)
        payables_turnover_days = st.number_input("Payables Turnover Days", step=0.01)

    if st.button("Predict Bankruptcy Risk"):
        try:
            manual_ratios = pd.DataFrame([[
                working_capital_ratio,
                roa,
                ebit_to_assets,
                debt_to_equity,
                interest_coverage,
                ocf_to_debt,
                receivables_turnover,
                payables_turnover_days
            ]], columns=[
                'working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
                'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days'
            ])

            scaled_manual = scaler.transform(manual_ratios)
            prediction_manual = model.predict_proba(scaled_manual)[0][1]

            pred_percent_manual = prediction_manual * 100

            fig_manual = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_percent_manual,
                title={'text': "Bankruptcy Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': prussian_red},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                }
            ))
            st.plotly_chart(fig_manual, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
