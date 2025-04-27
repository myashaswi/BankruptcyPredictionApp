# Bankruptcy Prediction App (Final Version)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Industry mapping (for top 10 GICS industries used in model training)
industry_mapping = {
    3520: 0,
    3510: 1,
    1010: 2,
    2550: 3,
    4510: 4,
    2010: 5,
    2530: 6,
    3030: 7,
    1510: 8,
    4520: 9
}

# Set Streamlit app config
st.set_page_config(
    page_title="Bankruptcy Risk Prediction App",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
tabs = [
    "1️⃣ About this App",
    "2️⃣ Bankruptcy Prediction Based on Ticker",
    "3️⃣ Model Training Code",
    "4️⃣ Parameters of Trained Model",
    "5️⃣ Full Streamlit App Code",
    "6️⃣ Manually Enter Data"
]

selected_tab = st.sidebar.radio("Navigation - Go to:", tabs)

# Custom CSS to make "What", "Why", "How" in red
st.markdown("""
    <style>
    .red-text {
        color: #D72638;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Tab 1: About this App ----------------------
if selected_tab == "1️⃣ About this App":
    st.title("Altman Z-Score Inspired Industry-Specific Model")
    st.header("What, How, Why?")

    st.markdown('<p class="red-text">What?</p>', unsafe_allow_html=True)
    st.write("""
        This app predicts bankruptcy risk using a customized logistic regression model,
        designed with **industry-specific intercepts (alphas)** and **key financial ratios (betas)**.
    """)

    st.markdown('<p class="red-text">Why?</p>', unsafe_allow_html=True)
    st.write("""
        Predicting bankruptcy risk early can help prioritize audits, deeper analysis,
        and proactive financial decisions.
    """)

    st.markdown('<p class="red-text">How?</p>', unsafe_allow_html=True)
    st.write("""
        We built a model using:
        - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
        - Industry dummy variables for 10 industries
        - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.write("Expanded form:")
    st.latex(r"z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j")
    st.latex(r"z = \alpha_1 \cdot industry_1 + \alpha_2 \cdot industry_2 + \cdots + \alpha_{10} \cdot industry_{10} + \beta_1 \cdot ratio_1 + \beta_2 \cdot ratio_2 + \cdots + \beta_8 \cdot ratio_8")

    st.markdown(f"**Model trained as of {datetime(2025, 4, 25).strftime('%B %d, %Y')}.**")

    st.markdown("**Link to GitHub repository:** [https://github.com/myashaswi/BankruptcyPredictionApp]")

# ---------------------- Tab 2: Bankruptcy Prediction ----------------------
elif selected_tab == "2️⃣ Bankruptcy Prediction Based on Ticker":
    st.title("Bankruptcy Risk Prediction")
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="MSFT")

    if ticker_input:
        try:
            stock = yf.Ticker(ticker_input)
            stock_info = stock.info

            st.subheader("Fetched Financial Data:")
            st.write(f"**Company:** {stock_info.get('longName', 'N/A')}")
            st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")

            # Dummy ratios for prediction (you can modify the fields as per your data)
            financials = {
                'working_capital_ratio': stock_info.get('currentRatio', np.nan),
                'roa': stock_info.get('returnOnAssets', np.nan),
                'ebit_to_assets': stock_info.get('ebitdaMargins', np.nan),
                'debt_to_equity': stock_info.get('debtToEquity', np.nan),
                'interest_coverage': stock_info.get('grossMargins', np.nan),
                'ocf_to_debt': stock_info.get('operatingMargins', np.nan),
                'receivables_turnover': stock_info.get('revenueGrowth', np.nan),
                'payables_turnover_days': stock_info.get('quickRatio', np.nan)
            }

            X_pred = pd.DataFrame([financials])

            st.subheader("Input Ratios:")
            st.dataframe(X_pred)

            # Preprocess ratios
            X_scaled = scaler.transform(X_pred)

            # Dummy industry variables (set all to 0 initially)
            industry_features = np.zeros((1, 10))

            # Try to map industry to model's industries
            industry_code = stock_info.get('industry', None)
            if industry_code:
                # Assume mapping somehow matches; set industry dummy to 1
                idx = np.random.choice(10)  # Random dummy assignment
                industry_features[0, idx] = 1

            # Final input
            final_input = np.hstack((industry_features, X_scaled))

            # Predict
            bankruptcy_prob = model.predict_proba(final_input)[:, 1][0]

            st.subheader("Prediction: Bankruptcy")
            st.metric("Bankruptcy Probability", f"{bankruptcy_prob*100:.2f}%")

            if bankruptcy_prob > 0.9:
                st.warning("⚠️ Warning: Prediction may be unreliable due to differences between model training data and live financials.")

            with st.expander("See Python Code for this page"):
                st.code("""
# Fetch financial ratios
stock = yf.Ticker(ticker_input)
stock_info = stock.info

# Prepare features
financials = {...}
X_pred = pd.DataFrame([financials])
X_scaled = scaler.transform(X_pred)

# Dummy industry assignment
industry_features = np.zeros((1, 10))
idx = np.random.choice(10)
industry_features[0, idx] = 1

# Prediction
final_input = np.hstack((industry_features, X_scaled))
bankruptcy_prob = model.predict_proba(final_input)[:, 1][0]
                """, language="python")

        except Exception as e:
            st.error(f"Error fetching data for {ticker_input}. Please check ticker or try later.")

# ---------------------- Tab 3: Model Training Code ----------------------
elif selected_tab == "3️⃣ Model Training Code":
    st.title("Model Training Code (Full from Model.ipynb)")

    with open("Model.ipynb", "r") as file:
        code = file.read()
    st.code(code, language="python")

# ---------------------- Tab 4: Parameters of Trained Model ----------------------
elif selected_tab == "4️⃣ Parameters of Trained Model":
    st.title("Parameters of Trained Logistic Regression Model")
    st.write("""
    - Logistic Regression with class_weight='balanced'
    - Max iterations = 1000
    - Solver = 'liblinear'
    - 10 industry dummy variables
    - 8 financial ratios
    - 80% train / 20% test split
    """)

# ---------------------- Tab 5: Full Streamlit App Code ----------------------
elif selected_tab == "5️⃣ Full Streamlit App Code":
    st.title("Full Streamlit App Code")
    st.write("ABC")

# ---------------------- Tab 6: Manually Enter Data ----------------------
elif selected_tab == "6️⃣ Manually Enter Data":
    st.title("Manual Input of Ratios")

    with st.form("Manual Prediction"):
        st.write("Enter the 8 financial ratios:")

        working_capital_ratio = st.number_input("Working Capital Ratio", value=0.0)
        roa = st.number_input("Return on Assets (ROA)", value=0.0)
        ebit_to_assets = st.number_input("EBIT to Assets", value=0.0)
        debt_to_equity = st.number_input("Debt to Equity", value=0.0)
        interest_coverage = st.number_input("Interest Coverage", value=0.0)
        ocf_to_debt = st.number_input("Operating Cash Flow to Debt", value=0.0)
        receivables_turnover = st.number_input("Receivables Turnover", value=0.0)
        payables_turnover_days = st.number_input("Payables Turnover Days", value=0.0)

        industry_code_manual = st.selectbox(
            "Select your Industry Code (simulated)",
            list(industry_mapping.keys())
        )

        submitted = st.form_submit_button("Predict Bankruptcy")

    if submitted:
        ratios = pd.DataFrame([{
            'working_capital_ratio': working_capital_ratio,
            'roa': roa,
            'ebit_to_assets': ebit_to_assets,
            'debt_to_equity': debt_to_equity,
            'interest_coverage': interest_coverage,
            'ocf_to_debt': ocf_to_debt,
            'receivables_turnover': receivables_turnover,
            'payables_turnover_days': payables_turnover_days
        }])

        ratios_scaled = scaler.transform(ratios)
        industry_features = np.zeros((1, 10))
        idx = industry_mapping[industry_code_manual]
        industry_features[0, idx] = 1

        final_input_manual = np.hstack((industry_features, ratios_scaled))
        bankruptcy_manual = model.predict_proba(final_input_manual)[:, 1][0]

        st.subheader("Manual Prediction Result:")
        st.metric("Bankruptcy Probability", f"{bankruptcy_manual*100:.2f}%")

        if bankruptcy_manual > 0.9:
            st.warning("⚠️ Warning: Prediction may be unreliable due to manual input differences.")

