import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
from datetime import datetime

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar navigation
st.sidebar.title('Navigation - Go to:')
page = st.sidebar.radio("", (
    "🔴 About this App",
    "📊 Bankruptcy Prediction Based on Ticker",
    "📋 Regression Output",
    "📑 Model Training Code",
    "📑 Streamlit App Code",
    "📑 Scaler Saving Code",
    "📑 Model Saving Code",
    "📑 Requirements.txt"
))

# Prussian Blue HEX code
PRUSSIAN_BLUE = '#003153'

# Page 1: About this App
if page == "🔴 About this App":
    st.title('Bankruptcy Prediction App')
    st.header('Altman Z-Score Inspired Industry-Specific Model')

    st.markdown(f'<h2 style="color:{PRUSSIAN_BLUE};">What, How, Why?</h2>', unsafe_allow_html=True)

    st.markdown(f'<h3 style="color:{PRUSSIAN_BLUE};">What?</h3>', unsafe_allow_html=True)
    st.write("This app predicts bankruptcy risk using a customized logistic regression model, designed with **industry-specific intercepts** (alphas) and **key financial ratios** (betas).")

    st.markdown(f'<h3 style="color:{PRUSSIAN_BLUE};">Why?</h3>', unsafe_allow_html=True)
    st.write("Predicting bankruptcy risk early can help prioritize audits, deeper analysis, and proactive financial decisions.")

    st.markdown(f'<h3 style="color:{PRUSSIAN_BLUE};">How?</h3>', unsafe_allow_html=True)
    st.write("We built a model using:")
    st.markdown("""
    - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
    - Industry dummy variables for 10 industries
    - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.markdown("### Expanded form:")
    st.latex(r"""z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j""")
    st.latex(r"""z = \alpha_1 \cdot industry_1 + \alpha_2 \cdot industry_2 + \cdots + \alpha_{10} \cdot industry_{10} + \beta_1 \cdot ratio_1 + \beta_2 \cdot ratio_2 + \cdots + \beta_8 \cdot ratio_8""")

# Page 2: Bankruptcy Prediction
elif page == "📊 Bankruptcy Prediction Based on Ticker":
    st.title('📊 Bankruptcy Risk Prediction')
    ticker = st.text_input(f"Enter Stock Ticker (e.g., AAPL, MSFT):", value="MSFT")

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            bs = stock.balance_sheet
            if bs.empty:
                st.warning("⚠️ Data not available for this ticker.")
            else:
                latest = bs.iloc[:, 0]
                total_assets = latest.get('Total Assets', np.nan)
                total_liab = latest.get('Total Liab', np.nan)
                total_debt = latest.get('Short Long Term Debt', np.nan) + latest.get('Long Term Debt', np.nan)
                net_income = stock.financials.get('Net Income', np.nan).iloc[0]
                ebit = stock.financials.get('EBIT', np.nan).iloc[0]
                interest_expense = stock.financials.get('Interest Expense', np.nan).iloc[0]
                ocf = stock.cashflow.get('Total Cash From Operating Activities', np.nan).iloc[0]
                revenue = stock.financials.get('Total Revenue', np.nan).iloc[0]
                
                # Derived ratios
                ratios = {
                    'working_capital_ratio': (total_assets - total_liab) / total_assets if total_assets else np.nan,
                    'roa': net_income / total_assets if total_assets else np.nan,
                    'ebit_to_assets': ebit / total_assets if total_assets else np.nan,
                    'debt_to_equity': total_debt / (total_assets - total_liab) if (total_assets - total_liab) else np.nan,
                    'interest_coverage': ebit / interest_expense if interest_expense else np.nan,
                    'ocf_to_debt': ocf / total_debt if total_debt else np.nan,
                    'receivables_turnover': revenue / latest.get('Net Receivables', np.nan) if latest.get('Net Receivables', np.nan) else np.nan,
                    'payables_turnover_days': latest.get('Accounts Payable', np.nan) / (revenue / 365) if revenue else np.nan
                }

                df_ratios = pd.DataFrame([ratios])
                df_ratios_scaled = scaler.transform(df_ratios)
                prediction = model.predict(df_ratios_scaled)[0]
                probability = model.predict_proba(df_ratios_scaled)[0][1]

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.error(f"⚠️ High bankruptcy risk! (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ Low bankruptcy risk (Probability: {probability:.2%})")

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Page 3: Regression Output
elif page == "📋 Regression Output":
    st.title("📋 Regression Model Summary")
    st.write("Model uses logistic regression with industry-specific intercepts and 8 financial ratios.")
    st.write("Coefficients and model performance shown below:")
    st.image("pages/files/regression_output.png", caption="Model Coefficients")

# Other Pages (Model Code, App Code, etc.)
else:
    st.title(page)
    st.write("Code block for:", page)
    st.info("Please refer to the GitHub repo or full project documentation for code details.")

    if "Training" in page:
        st.code("""
# Model training code snippet
from sklearn.linear_model import LogisticRegression
...
""", language='python')
    elif "Streamlit App" in page:
        st.code("""
# Streamlit dashboard app code
import streamlit as st
...
""", language='python')
    elif "Scaler" in page:
        st.code("""
# Save Scaler
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'scaler.pkl')
""", language='python')
    elif "Model Saving" in page:
        st.code("""
# Save Model
joblib.dump(model, 'model.pkl')
""", language='python')
    elif "Requirements" in page:
        st.code("""
streamlit
scikit-learn
yfinance
pandas
numpy
joblib
""", language='text')
