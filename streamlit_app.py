import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import joblib

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define financial ratios to use
features = [
    'working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
    'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days'
]

# Set theme colors
RED = "#c95c5d"

# Sidebar Navigation
st.sidebar.title("Navigation - Go to:")
page = st.sidebar.radio(
    "",
    [
        "1 About this App",
        "2 Bankruptcy Prediction Based on Ticker",
        "3 Model Training Code",
        "4 Parameters of Trained Model",
        "5 Full Streamlit App Code",
        "6 Manually Enter Data"
    ]
)

# 1. About this App
if page.startswith("1"):
    st.title("Altman Z-Score Inspired Industry-Specific Model")
    st.markdown(f"<h2 style='color:{RED};'>What, How, Why?</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='color:{RED};'>What?</h3>", unsafe_allow_html=True)
    st.write(
        "This app predicts bankruptcy risk using a customized logistic regression model, "
        "designed with **industry-specific intercepts** (alphas) and **key financial ratios** (betas)."
    )

    st.markdown(f"<h3 style='color:{RED};'>Why?</h3>", unsafe_allow_html=True)
    st.write(
        "Predicting bankruptcy risk early can help prioritize audits, deeper analysis, and proactive financial decisions."
    )

    st.markdown(f"<h3 style='color:{RED};'>How?</h3>", unsafe_allow_html=True)
    st.write("We built a model using:")
    st.markdown("""
    - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
    - Industry dummy variables for 10 industries
    - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.subheader("Expanded form:")
    st.latex(r'''
    z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j
    ''')
    st.latex(r'''
    z = \alpha_1 \cdot industry_1 + \alpha_2 \cdot industry_2 + \cdots + \alpha_{10} \cdot industry_{10}
    + \beta_1 \cdot ratio_1 + \beta_2 \cdot ratio_2 + \cdots + \beta_8 \cdot ratio_8
    ''')

    st.write("**Model trained as of April 25, 2025.**")

    st.markdown("[Link to GitHub Repository](https://github.com/myashaswi/BankruptcyPredictionApp/)", unsafe_allow_html=True)

# 2. Bankruptcy Prediction Based on Ticker
elif page.startswith("2"):
    st.title("Bankruptcy Risk Prediction")
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="MSFT")

    if ticker_input:
        try:
            ticker = yf.Ticker(ticker_input)
            info = ticker.info

            company_name = info.get("longName", "N/A")
            industry_name = info.get("industry", "N/A")

            st.subheader("Fetched Financial Data:")
            st.write(f"**Company:** {company_name}")
            st.write(f"**Industry:** {industry_name}")

            # Get Financial Ratios
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow

            if financials.empty or balance_sheet.empty or cashflow.empty:
                st.error(f"Error fetching complete financials for {ticker_input}. Please try another stock or later.")
            else:
                # Calculate Ratios (simple example: assume latest available data)
                try:
                    working_capital_ratio = (balance_sheet.loc["Total Current Assets"][0] - balance_sheet.loc["Total Current Liabilities"][0]) / balance_sheet.loc["Total Assets"][0]
                    roa = financials.loc["Net Income"][0] / balance_sheet.loc["Total Assets"][0]
                    ebit_to_assets = financials.loc["EBIT"][0] / balance_sheet.loc["Total Assets"][0]
                    debt_to_equity = balance_sheet.loc["Total Liabilities Net Minority Interest"][0] / balance_sheet.loc["Stockholders Equity"][0]
                    interest_coverage = financials.loc["EBIT"][0] / financials.loc["Interest Expense"][0]
                    ocf_to_debt = cashflow.loc["Total Cash From Operating Activities"][0] / balance_sheet.loc["Total Liabilities Net Minority Interest"][0]
                    receivables_turnover = financials.loc["Total Revenue"][0] / balance_sheet.loc["Net Receivables"][0]
                    payables_turnover_days = (balance_sheet.loc["Accounts Payable"][0] / financials.loc["Cost Of Revenue"][0]) * 365

                    input_df = pd.DataFrame([[
                        working_capital_ratio, roa, ebit_to_assets, debt_to_equity,
                        interest_coverage, ocf_to_debt, receivables_turnover, payables_turnover_days
                    ]], columns=features)

                    st.subheader("Input Ratios:")
                    st.dataframe(input_df.style.format(precision=4))

                    # Predict
                    scaled_input = scaler.transform(input_df)
                    bankruptcy_probability = model.predict_proba(scaled_input)[0][1]

                    prediction = "Bankrupt" if bankruptcy_probability >= 0.5 else "Not Bankrupt"
                    st.subheader(f"Prediction: {prediction}")
                    st.metric(label="Bankruptcy Probability", value=f"{bankruptcy_probability*100:.2f}%")

                    # Show Code
                    with st.expander("See Python Code for this page"):
                        st.code("""
# Fetch financial statements
ticker = yf.Ticker(ticker_input)
financials = ticker.financials
balance_sheet = ticker.balance_sheet
cashflow = ticker.cashflow

# Calculate financial ratios
working_capital_ratio = (balance_sheet.loc["Total Current Assets"][0] - balance_sheet.loc["Total Current Liabilities"][0]) / balance_sheet.loc["Total Assets"][0]
roa = financials.loc["Net Income"][0] / balance_sheet.loc["Total Assets"][0]
ebit_to_assets = financials.loc["EBIT"][0] / balance_sheet.loc["Total Assets"][0]
debt_to_equity = balance_sheet.loc["Total Liabilities Net Minority Interest"][0] / balance_sheet.loc["Stockholders Equity"][0]
interest_coverage = financials.loc["EBIT"][0] / financials.loc["Interest Expense"][0]
ocf_to_debt = cashflow.loc["Total Cash From Operating Activities"][0] / balance_sheet.loc["Total Liabilities Net Minority Interest"][0]
receivables_turnover = financials.loc["Total Revenue"][0] / balance_sheet.loc["Net Receivables"][0]
payables_turnover_days = (balance_sheet.loc["Accounts Payable"][0] / financials.loc["Cost Of Revenue"][0]) * 365

# Scale ratios and predict bankruptcy
scaled_input = scaler.transform(input_df)
bankruptcy_probability = model.predict_proba(scaled_input)[0][1]
prediction = "Bankrupt" if bankruptcy_probability >= 0.5 else "Not Bankrupt"
                        """, language="python")

                except Exception as e:
                    st.error(f"Error processing financial ratios for {ticker_input}. Details: {e}")
        except Exception as e:
            st.error(f"Error fetching data for {ticker_input}. Please check ticker or try later.")

# 3. Model Training Code
elif page.startswith("3"):
    st.title("Model Training Code")
    st.markdown("Here is the full code used to train the model:")

    with open("Model.ipynb", "r") as file:
        content = file.read()
    st.code(content)

# 4. Parameters of Trained Model
elif page.startswith("4"):
    st.title("Parameters of the Trained Model")
    st.write("This app uses logistic regression with industry-specific intercepts and financial ratios.")

# 5. Full Streamlit App Code
elif page.startswith("5"):
    st.title("Full Streamlit App Code")

    with open("streamlit_app.py", "r") as f:
        code = f.read()
    st.code(code, language="python")

# 6. Manually Enter Data
elif page.startswith("6"):
    st.title("Manual Data Entry")
    st.write("You can manually enter your financial ratios here:")

    manual_data = {}
    for feature in features:
        manual_data[feature] = st.number_input(f"Enter {feature}:", step=0.01)

    if st.button("Predict Bankruptcy"):
        input_df = pd.DataFrame([manual_data])
        scaled_input = scaler.transform(input_df)
        bankruptcy_probability = model.predict_proba(scaled_input)[0][1]

        prediction = "Bankrupt" if bankruptcy_probability >= 0.5 else "Not Bankrupt"
        st.subheader(f"Prediction: {prediction}")
        st.metric(label="Bankruptcy Probability", value=f"{bankruptcy_probability*100:.2f}%")
