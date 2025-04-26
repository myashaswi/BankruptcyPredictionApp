# Bankruptcy Prediction App - Final Streamlit Code
# (BankruptcyPredictionApp2 repository)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime

# Load Model and Scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define Features
features = [
    'working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
    'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days'
]

# Page Config
st.set_page_config(page_title='Bankruptcy Prediction App', layout='wide', initial_sidebar_state='expanded')

# Sidebar Navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", [
    '📚 About this App',
    '🔎 Bankruptcy Prediction'
])

# About this App Page
if page == '📚 About this App':
    st.title('Bankruptcy Prediction App')
    st.header('Altman Z-Score Model Adaptation')

    st.markdown("""
    ### What, How, Why?

    **What?**
    This app uses an adaptation of the Altman Z-Score model trained on financial ratios to predict potential bankruptcy risk over a 5-year horizon.

    **Why?**
    Early warning of bankruptcy risk can help investors, auditors, and management teams prioritize deeper investigation and financial health assessments.

    **How?**
    Using a logistic regression model trained on Compustat data from 2013-2023, across 26 industries. Predictions are based on 8 standardized financial ratios commonly associated with firm stability.

    - **Training Horizon:** 7 years
    - **Testing Horizon:** 3 years
    - **Prediction Horizon:** 5 years ahead bankruptcy risk

    --
    **GitHub Repository**: [Link to this App's Repository](https://github.com/your-username/BankruptcyPredictionApp2)
    """)

# Bankruptcy Prediction Page
elif page == '🔎 Bankruptcy Prediction':
    st.title('Bankruptcy Prediction for a Public Company')

    st.markdown("""
    ### Instructions:
    1. Enter a valid stock ticker (example: AAPL, MSFT, TSLA).
    2. App will pull recent financial data from Yahoo Finance.
    3. App will compute financial ratios and predict bankruptcy likelihood.
    """)

    ticker = st.text_input('Enter Stock Ticker:', value='AAPL')

    if st.button('Predict Bankruptcy Risk'):
        try:
            ticker_data = yf.Ticker(ticker)
            bs = ticker_data.balance_sheet
            is_data = ticker_data.financials
            cf = ticker_data.cashflow

            # Check if all necessary data is available
            required_bs_items = ['Total Current Assets', 'Total Current Liabilities', 'Total Assets', 'Total Debt']
            required_is_items = ['Net Income', 'EBIT', 'Total Revenue']
            required_cf_items = ['Total Cash From Operating Activities']

            # Extract required items safely
            def get_value(df, item):
                try:
                    return df.loc[item].iloc[0]
                except:
                    return np.nan

            data_dict = {
                'Total Current Assets': get_value(bs, 'Total Current Assets'),
                'Total Current Liabilities': get_value(bs, 'Total Current Liabilities'),
                'Total Assets': get_value(bs, 'Total Assets'),
                'Total Debt': get_value(bs, 'Total Debt'),
                'Net Income': get_value(is_data, 'Net Income'),
                'EBIT': get_value(is_data, 'EBIT'),
                'Total Revenue': get_value(is_data, 'Total Revenue'),
                'Operating Cash Flow': get_value(cf, 'Total Cash From Operating Activities')
            }

            df_input = pd.DataFrame([data_dict])

            if df_input.isnull().sum().sum() > 0:
                st.warning('⚠️ Some financial data missing. Prediction may not be reliable.')

            # Calculate Ratios
            df_input['working_capital_ratio'] = (df_input['Total Current Assets'] - df_input['Total Current Liabilities']) / df_input['Total Assets']
            df_input['roa'] = df_input['Net Income'] / df_input['Total Assets']
            df_input['ebit_to_assets'] = df_input['EBIT'] / df_input['Total Assets']
            df_input['debt_to_equity'] = df_input['Total Debt'] / (df_input['Total Assets'] - df_input['Total Debt'])
            df_input['interest_coverage'] = df_input['EBIT'] / (0.05 * df_input['Total Debt'])  # approximate
            df_input['ocf_to_debt'] = df_input['Operating Cash Flow'] / df_input['Total Debt']
            df_input['receivables_turnover'] = df_input['Total Revenue'] / (0.2 * df_input['Total Current Assets'])  # approximate
            df_input['payables_turnover_days'] = (df_input['Total Current Liabilities'] / (0.6 * df_input['Total Revenue'])) * 365  # approximate

            # Select and scale features
            X_input = df_input[features]
            X_scaled = scaler.transform(X_input)

            # Predict
            prediction = model.predict(X_scaled)
            probability = model.predict_proba(X_scaled)[:, 1][0]

            st.success(f'✅ Prediction completed for {ticker.upper()}')
            st.metric('Bankruptcy Probability (5 Years)', f'{probability:.2%}')

            if prediction[0] == 1:
                st.error('⚠️ High Risk of Bankruptcy (Model Prediction: Bankrupt)')
            else:
                st.success('✅ Low Risk of Bankruptcy (Model Prediction: Not Bankrupt)')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
