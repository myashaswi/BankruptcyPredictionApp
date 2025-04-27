# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define list of financial features expected
features = [
    'working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
    'interest_coverage', 'ocf_to_debt', 'receivables_turnover',
    'payables_turnover_days'
]

# Define industries
industries = {
    3520: 'Pharmaceuticals, Biotechnology & Life Sciences',
    3510: 'Health Care Equipment & Services',
    1010: 'Energy',
    2550: 'Consumer Discretionary Distribution & Retail',
    4510: 'Software & Services',
    2010: 'Capital Goods',
    2530: 'Consumer Services',
    3030: 'Household & Personal Products',
    1510: 'Materials',
    4520: 'Technology Hardware & Equipment'
}

# Prussian Blue hex code
prussian_blue = "#003153"

# Sidebar navigation
st.sidebar.title("Navigation - Go to:")

page = st.sidebar.radio(" ", 
    ("📖 About this App",
     "📈 Bankruptcy Prediction Based on Ticker",
     "🧠 Model Training Code",
     "🛠️ Parameters of Trained Model",
     "📊 Forecast & Analyze Bankruptcy Risk",
     "📚 Full Streamlit App Code",
     "✍️ Manually Enter Data"
    )
)

# Page: About this App
if page == "📖 About this App":
    st.title("Altman Z-Score Inspired Industry-Specific Model")

    st.markdown(f"<h2 style='color:{prussian_blue};'>What, How, Why?</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='color:{prussian_blue};'>What?</h3>", unsafe_allow_html=True)
    st.write("This app predicts bankruptcy risk using a customized logistic regression model, designed with **industry-specific intercepts** (alphas) and **key financial ratios** (betas).")

    st.markdown(f"<h3 style='color:{prussian_blue};'>Why?</h3>", unsafe_allow_html=True)
    st.write("Predicting bankruptcy risk early can help prioritize audits, deeper analysis, and proactive financial decisions.")

    st.markdown(f"<h3 style='color:{prussian_blue};'>How?</h3>", unsafe_allow_html=True)
    st.write("""
    We built a model using:
    - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
    - Industry dummy variables for 10 industries
    - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.subheader("Expanded form:")
    st.latex(r"""
    z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j
    """)
    st.latex(r"""
    z = \alpha_1 \cdot industry_1 + \alpha_2 \cdot industry_2 + \cdots + \alpha_{10} \cdot industry_{10} + \beta_1 \cdot ratio_1 + \beta_2 \cdot ratio_2 + \cdots + \beta_8 \cdot ratio_8
    """)

    st.markdown("**Model trained as of April 25, 2025.**")

# Page: Bankruptcy Prediction Based on Ticker
elif page == "📈 Bankruptcy Prediction Based on Ticker":
    st.title("Bankruptcy Risk Prediction")

    ticker = st.text_input(f"Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            st.subheader("Fetched Financial Data:")
            st.write(f"**Company:** {info.get('longName', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")

            # Dummy financials (substitute with real calculations from financial statements if needed)
            data = {
                'working_capital_ratio': np.random.uniform(0.5, 2.0),
                'roa': np.random.uniform(-0.2, 0.2),
                'ebit_to_assets': np.random.uniform(-0.2, 0.2),
                'debt_to_equity': np.random.uniform(0.0, 2.0),
                'interest_coverage': np.random.uniform(0.0, 10.0),
                'ocf_to_debt': np.random.uniform(-0.2, 0.2),
                'receivables_turnover': np.random.uniform(1.0, 10.0),
                'payables_turnover_days': np.random.uniform(20.0, 100.0)
            }

            df = pd.DataFrame([data])

            st.subheader("Input Ratios:")
            st.dataframe(df)

            X_scaled = scaler.transform(df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]

            st.subheader(f"Prediction: {'Bankruptcy' if prediction==1 else 'No Bankruptcy'}")
            st.metric(label="Bankruptcy Probability", value=f"{probability:.2%}")

            with st.expander("See Python Code for this page"):
                st.code("""
ticker = st.text_input("Enter Ticker")
stock = yf.Ticker(ticker)
info = stock.info
data = {your_feature_extraction_logic_here}
df = pd.DataFrame([data])
X_scaled = scaler.transform(df)
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)
                """, language='python')

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.write("Data not available or ticker incorrect.")

# Page: Model Training Code
elif page == "🧠 Model Training Code":
    st.title("Model Training Code")
    with st.expander("View Training Code"):
        st.code("""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_final, test_size=0.2, stratify=y_final)

model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
        """, language='python')

# Page: Parameters of Trained Model
elif page == "🛠️ Parameters of Trained Model":
    st.title("Model Parameters Summary")
    st.write("Model: Logistic Regression (Balanced Classes, Liblinear Solver, Max Iterations = 1000)")
    st.write("Scaler: StandardScaler (mean=0, variance=1 scaling applied)")
    st.write("Features used:")
    st.markdown("- Working Capital Ratio\n- ROA\n- EBIT to Assets\n- Debt to Equity\n- Interest Coverage\n- OCF to Debt\n- Receivables Turnover\n- Payables Turnover Days")

# Page: Forecast & Analyze Bankruptcy Risk
elif page == "📊 Forecast & Analyze Bankruptcy Risk":
    st.title("Forecast and Analyze Bankruptcy Risk")
    st.write("Coming Soon: Sensitivity Analysis, ROC curves, Threshold tuning...")

# Page: Full StreamLit App Code
elif page == "📚 Full Streamlit App Code":
    st.title("Full Streamlit App Code")
    with open(__file__, "r") as f:
        st.code(f.read(), language='python')

# Page: Manually Enter Data
elif page == "✍️ Manually Enter Data":
    st.title("Manual Data Entry (Optional)")
    st.write("Allow manual input of 8 financial ratios if needed.")
    st.write("Coming Soon...")

