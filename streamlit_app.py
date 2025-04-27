import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pickle
import joblib

# Load model and scaler
model = pickle.load(open("model (1).pkl", "rb"))
scaler = joblib.load("scaler (1).pkl")

# Set page configuration
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("Navigation - Go to:", [
    "1 About this App",
    "2 Bankruptcy Prediction Based on Ticker",
    "3 Model Training Code",
    "4 Full Streamlit App Code",
    "5 Manually Enter Data"
])

prussian_red = "#c95c5d"

# ========== PAGE 1: About this App ==========
if page.startswith("1"):
    st.title("Altman Z-Score Inspired Industry-Specific Model")
    st.markdown(f"<h2 style='color:{prussian_red};'>What, How, Why?</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='color:{prussian_red};'>What?</h3>", unsafe_allow_html=True)
    st.write("This app predicts bankruptcy risk using a customized logistic regression model, with **industry-specific intercepts** and **key financial ratios**.")

    st.markdown(f"<h3 style='color:{prussian_red};'>Why?</h3>", unsafe_allow_html=True)
    st.write("Early prediction of bankruptcy risk helps prioritize audits, deeper analysis, and proactive financial decisions.")

    st.markdown(f"<h3 style='color:{prussian_red};'>How?</h3>", unsafe_allow_html=True)
    st.write("""
    - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
    - Industry dummy variables for 10 industries
    - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.latex(r"""z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j""")

    st.write("Model trained as of **April 25, 2025**.")
    st.markdown("---")
    st.markdown("🔗 [GitHub Repository for this App](https://your-github-link-here)")

# ========== PAGE 2: Bankruptcy Prediction Based on Ticker ==========
elif page.startswith("2"):
    st.title("Bankruptcy Risk Prediction Based on Stock Ticker")

    ticker = st.text_input("Enter Stock Ticker (example: AAPL, MSFT, NVDA)").upper()

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fin = stock.financials
            bs = stock.balance_sheet
            cf = stock.cashflow

            # Soft Fetch ratios
            ratios = {}

            try:
                ratios['working_capital_ratio'] = (bs.loc['Total Current Assets'][0] - bs.loc['Total Current Liabilities'][0]) / bs.loc['Total Assets'][0]
            except:
                ratios['working_capital_ratio'] = 0

            try:
                ratios['roa'] = fin.loc['Net Income'][0] / bs.loc['Total Assets'][0]
            except:
                ratios['roa'] = 0

            try:
                ratios['ebit_to_assets'] = fin.loc['EBIT'][0] / bs.loc['Total Assets'][0]
            except:
                ratios['ebit_to_assets'] = 0

            try:
                ratios['debt_to_equity'] = bs.loc['Total Debt'][0] / (bs.loc['Total Assets'][0] - bs.loc['Total Debt'][0])
            except:
                ratios['debt_to_equity'] = 0

            try:
                ratios['interest_coverage'] = fin.loc['EBIT'][0] / (fin.loc['Interest Expense'][0])
            except:
                ratios['interest_coverage'] = 0

            try:
                ratios['ocf_to_debt'] = cf.loc['Total Cash From Operating Activities'][0] / bs.loc['Total Debt'][0]
            except:
                ratios['ocf_to_debt'] = 0

            try:
                ratios['receivables_turnover'] = fin.loc['Total Revenue'][0] / bs.loc['Accounts Receivable'][0]
            except:
                ratios['receivables_turnover'] = 0

            try:
                ratios['payables_turnover_days'] = (bs.loc['Accounts Payable'][0] / fin.loc['Cost Of Revenue'][0]) * 365
            except:
                ratios['payables_turnover_days'] = 0

            input_df = pd.DataFrame([ratios])

            st.subheader("Input Ratios Used:")
            st.dataframe(input_df)

            scaled = scaler.transform(input_df)
            pred = model.predict_proba(scaled)[0][1]
            pred_percent = pred * 100

            # Plot Gauge Chart
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
            st.error(f"Failed to fetch data or predict. Error: {e}")

# ========== PAGE 3: Model Training Code ==========
elif page.startswith("3"):
    st.title("Model Training Code")

    try:
        with open("Model.ipynb", "r") as f:
            content = f.read()
        st.code(content, language="json")  # Model.ipynb is JSON structured
    except Exception as e:
        st.error(f"Could not load Model.ipynb. Error: {e}")

# ========== PAGE 4: Full Streamlit App Code ==========
elif page.startswith("4"):
    st.title("Full Streamlit App Code")

    try:
        with open("streamlit_app.py", "r") as f:
            code = f.read()
        st.code(code, language="python")
    except Exception as e:
        st.error(f"Could not load streamlit_app.py. Error: {e}")

# ========== PAGE 5: Manually Enter Data ==========
elif page.startswith("5"):
    st.title("Manual Data Entry for Bankruptcy Prediction")

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
                working_capital_ratio, roa, ebit_to_assets, debt_to_equity,
                interest_coverage, ocf_to_debt, receivables_turnover, payables_turnover_days
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
