import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib

# First, add these functions at the top of your file
def get_value(df, possible_keys):
    """Try to get a value from a dataframe using multiple possible keys"""
    if df is None or df.empty:
        return None
    
    for key in possible_keys:
        try:
            if key in df.index:
                value = df.loc[key][0]  # Get most recent value
                if pd.notnull(value) and value != 0:
                    return value
        except:
            continue
    
    return None

def calculate_financial_ratios(ticker):
    """Calculate all financial ratios with multiple fallback methods"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get different timeframes of financial data
        fin_annual = stock.financials
        fin_quarterly = stock.quarterly_financials
        
        bs_annual = stock.balance_sheet
        bs_quarterly = stock.quarterly_balance_sheet
        
        cf_annual = stock.cashflow
        cf_quarterly = stock.quarterly_cashflow
        
        # Use most recent data (quarterly if available, otherwise annual)
        fin = fin_quarterly if not fin_quarterly.empty else fin_annual
        bs = bs_quarterly if not bs_quarterly.empty else bs_annual
        cf = cf_quarterly if not cf_quarterly.empty else cf_annual
        
        # Display company information
        st.subheader(f"Company: {info.get('shortName', ticker)}")
        st.write(f"Industry: {info.get('industry', 'Unknown')}")
        st.write(f"Sector: {info.get('sector', 'Unknown')}")
        
        # Dictionary to store the calculated ratios
        ratios = {}
        
        # 1. Working Capital Ratio
        try:
            # Method 1: Standard calculation
            current_assets = get_value(bs, ['Total Current Assets', 'Current Assets'])
            current_liabilities = get_value(bs, ['Total Current Liabilities', 'Current Liabilities'])
            total_assets = get_value(bs, ['Total Assets'])
            
            if current_assets and current_liabilities and total_assets:
                ratios['working_capital_ratio'] = float((current_assets - current_liabilities) / total_assets)
            else:
                # Method 2: Alternative calculation using components
                cash = get_value(bs, ['Cash And Cash Equivalents', 'Cash', 'Cash And Short Term Investments'])
                receivables = get_value(bs, ['Accounts Receivable', 'Net Receivables'])
                inventory = get_value(bs, ['Inventory', 'Inventories'])
                payables = get_value(bs, ['Accounts Payable', 'Payables'])
                
                # If we have some components, make an estimate
                if total_assets and (cash or receivables or inventory) and payables:
                    current_assets_est = sum(filter(None, [cash, receivables, inventory]))
                    ratios['working_capital_ratio'] = float((current_assets_est - payables) / total_assets)
                else:
                    ratios['working_capital_ratio'] = 0.0
        except Exception as e:
            st.write(f"Working capital ratio calculation error: {e}")
            ratios['working_capital_ratio'] = 0.0
        
        # 2. Return on Assets (ROA)
        try:
            net_income = get_value(fin, ['Net Income', 'Net Income Common Stockholders'])
            total_assets = get_value(bs, ['Total Assets'])
            
            if net_income and total_assets:
                ratios['roa'] = float(net_income / total_assets)
            else:
                ratios['roa'] = 0.0
        except Exception as e:
            st.write(f"ROA calculation error: {e}")
            ratios['roa'] = 0.0
        
        # 3. EBIT to Assets
        try:
            # Method 1: Direct EBIT from financials
            ebit = get_value(fin, ['EBIT', 'Operating Income'])
            total_assets = get_value(bs, ['Total Assets'])
            
            if ebit and total_assets:
                ratios['ebit_to_assets'] = float(ebit / total_assets)
            else:
                # Method 2: Calculate EBIT from components
                net_income = get_value(fin, ['Net Income', 'Net Income Common Stockholders'])
                interest_expense = get_value(fin, ['Interest Expense', 'Interest And Debt Expense'])
                income_tax = get_value(fin, ['Income Tax Expense', 'Tax Provision'])
                
                if net_income and total_assets:
                    calculated_ebit = net_income
                    if interest_expense:
                        calculated_ebit += interest_expense
                    if income_tax:
                        calculated_ebit += income_tax
                    
                    ratios['ebit_to_assets'] = float(calculated_ebit / total_assets)
                else:
                    ratios['ebit_to_assets'] = 0.0
        except Exception as e:
            st.write(f"EBIT to assets calculation error: {e}")
            ratios['ebit_to_assets'] = 0.0
        
        # 4. Debt to Equity
        try:
            # Method 1: Simple calculation
            total_debt = get_value(bs, ['Total Debt', 'Long Term Debt', 'Short Long Term Debt'])
            
            # Calculate equity as total assets minus total debt
            total_assets = get_value(bs, ['Total Assets'])
            
            if total_debt and total_assets and total_assets > total_debt:
                equity = total_assets - total_debt
                ratios['debt_to_equity'] = float(total_debt / equity)
            else:
                # Method 2: Alternative equity calculation
                total_equity = get_value(bs, ['Total Stockholder Equity', 'Stockholders Equity'])
                
                if total_debt and total_equity and total_equity > 0:
                    ratios['debt_to_equity'] = float(total_debt / total_equity)
                else:
                    ratios['debt_to_equity'] = 0.0
        except Exception as e:
            st.write(f"Debt to equity calculation error: {e}")
            ratios['debt_to_equity'] = 0.0
        
        # 5. Interest Coverage
        try:
            ebit = get_value(fin, ['EBIT', 'Operating Income'])
            interest_expense = get_value(fin, ['Interest Expense', 'Interest And Debt Expense'])
            
            if ebit and interest_expense and interest_expense != 0:
                ratios['interest_coverage'] = float(ebit / interest_expense)
            else:
                # Alternative: Estimate from components if EBIT is available
                if ebit:
                    # If we have EBIT but no interest expense, assume strong coverage
                    ratios['interest_coverage'] = 10.0  # A high default value
                else:
                    ratios['interest_coverage'] = 0.0
        except Exception as e:
            st.write(f"Interest coverage calculation error: {e}")
            ratios['interest_coverage'] = 0.0
        
        # 6. Operating Cash Flow to Debt
        try:
            operating_cash_flow = get_value(cf, ['Total Cash From Operating Activities', 'Cash From Operating Activities'])
            total_debt = get_value(bs, ['Total Debt', 'Long Term Debt', 'Short Long Term Debt'])
            
            if operating_cash_flow and total_debt and total_debt != 0:
                ratios['ocf_to_debt'] = float(operating_cash_flow / total_debt)
            else:
                # Alternative calculation if total debt is missing
                total_liabilities = get_value(bs, ['Total Liabilities', 'Total Liabilities Net Minority Interest'])
                
                if operating_cash_flow and total_liabilities and total_liabilities != 0:
                    ratios['ocf_to_debt'] = float(operating_cash_flow / total_liabilities)
                else:
                    ratios['ocf_to_debt'] = 0.0
        except Exception as e:
            st.write(f"OCF to debt calculation error: {e}")
            ratios['ocf_to_debt'] = 0.0
        
        # 7. Receivables Turnover
        try:
            total_revenue = get_value(fin, ['Total Revenue', 'Revenue'])
            accounts_receivable = get_value(bs, ['Accounts Receivable', 'Net Receivables'])
            
            if total_revenue and accounts_receivable and accounts_receivable != 0:
                ratios['receivables_turnover'] = float(total_revenue / accounts_receivable)
            else:
                # If accounts receivable is missing, estimate from total assets
                if total_revenue and total_assets:
                    # Assume receivables are ~10% of total assets as a rough estimate
                    est_receivables = total_assets * 0.1
                    ratios['receivables_turnover'] = float(total_revenue / est_receivables)
                else:
                    ratios['receivables_turnover'] = 0.0
        except Exception as e:
            st.write(f"Receivables turnover calculation error: {e}")
            ratios['receivables_turnover'] = 0.0
        
        # 8. Payables Turnover Days
        try:
            accounts_payable = get_value(bs, ['Accounts Payable', 'Payables'])
            cost_of_revenue = get_value(fin, ['Cost Of Revenue', 'Cost of Revenue', 'COGS'])
            
            if accounts_payable and cost_of_revenue and cost_of_revenue != 0:
                ratios['payables_turnover_days'] = float((accounts_payable / cost_of_revenue) * 365)
            else:
                # Alternative calculation using operating expenses
                operating_expenses = get_value(fin, ['Operating Expense', 'Total Operating Expenses'])
                
                if accounts_payable and operating_expenses and operating_expenses != 0:
                    ratios['payables_turnover_days'] = float((accounts_payable / operating_expenses) * 365)
                else:
                    ratios['payables_turnover_days'] = 60.0  # Industry average default
        except Exception as e:
            st.write(f"Payables turnover days calculation error: {e}")
            ratios['payables_turnover_days'] = 60.0  # Industry average default
        
        # Hide the detailed calculation error messages
        if st.checkbox("Show detailed calculation errors", value=False):
            st.write("Calculation error messages will appear here.")
        else:
            # Clear any previous error messages
            placeholder = st.empty()
        
        return ratios, info
    
    except Exception as e:
        st.error(f"Failed to get financial data for {ticker}: {e}")
        return {}, {}

# Load model and scaler using joblib
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None
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

red = "#c95c5d"

# ========== PAGE 1: About this App ==========
if page.startswith("1"):
    st.title("Altman Z-Score Inspired Industry-Specific Model")
    st.markdown(f"<h2 style='color:{red};'>What, How, Why?</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='color:{red};'>What?</h3>", unsafe_allow_html=True)
    st.write("This app predicts bankruptcy risk using a customized logistic regression model, with **industry-specific intercepts** and **key financial ratios**.")

    st.markdown(f"<h3 style='color:{red};'>Why?</h3>", unsafe_allow_html=True)
    st.write("Early prediction of bankruptcy risk helps prioritize audits, deeper analysis, and proactive financial decisions.")

    st.markdown(f"<h3 style='color:{red};'>How?</h3>", unsafe_allow_html=True)
    st.write("""
    - Financial ratios like ROA, Debt-to-Equity, Interest Coverage, etc.
    - Industry dummy variables for 10 industries
    - Logistic regression combining both features to predict 5-year bankruptcy likelihood
    """)

    st.latex(r"""z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j""")

    st.write("Model trained as of **April 25, 2025**.")
    st.markdown("---")
    st.markdown("🔗 [GitHub Repository for this App](https://github.com/myashaswi/BankruptcyPredictionApp)")

# ========== PAGE 2: Bankruptcy Prediction Based on Ticker ==========
elif page.startswith("2"):
    st.title("Bankruptcy Risk Prediction Based on Stock Ticker")

    ticker = st.text_input("Enter Stock Ticker (example: AAPL, MSFT etc.)").upper()

    if ticker and model is not None and scaler is not None:
        try:
            st.write(f"Fetching data for {ticker}...")
            
            # Get basic company info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Display company name and industry
            company_name = info.get('longName', 'Unknown Company')
            industry = info.get('industry', 'Unknown Industry')
            st.write(f"**Company Name:** {company_name}")
            st.write(f"**Industry:** {industry}")
            
            # Get financial data
            fin = stock.financials
            bs = stock.balance_sheet
            cf = stock.cashflow
            
            # Create a dictionary for ratios
            ratios = {}
            
            # Calculate ratios with robust error handling
            # Working Capital Ratio
            try:
                current_assets = bs.loc['Total Current Assets'][0] if 'Total Current Assets' in bs.index else None
                current_liabilities = bs.loc['Total Current Liabilities'][0] if 'Total Current Liabilities' in bs.index else None
                total_assets = bs.loc['Total Assets'][0] if 'Total Assets' in bs.index else None
                
                if current_assets and current_liabilities and total_assets:
                    ratios['working_capital_ratio'] = float((current_assets - current_liabilities) / total_assets)
                else:
                    ratios['working_capital_ratio'] = 0.0
            except Exception as e:
                ratios['working_capital_ratio'] = 0.0
            
            # ROA
            try:
                net_income = fin.loc['Net Income'][0] if 'Net Income' in fin.index else None
                total_assets = bs.loc['Total Assets'][0] if 'Total Assets' in bs.index else None
                
                if net_income and total_assets:
                    ratios['roa'] = float(net_income / total_assets)
                else:
                    ratios['roa'] = 0.0
            except Exception as e:
                ratios['roa'] = 0.0
            
            # EBIT to Assets
            try:
                ebit = fin.loc['EBIT'][0] if 'EBIT' in fin.index else (fin.loc['Operating Income'][0] if 'Operating Income' in fin.index else None)
                total_assets = bs.loc['Total Assets'][0] if 'Total Assets' in bs.index else None
                
                if ebit and total_assets:
                    ratios['ebit_to_assets'] = float(ebit / total_assets)
                else:
                    ratios['ebit_to_assets'] = 0.0
            except Exception as e:
                ratios['ebit_to_assets'] = 0.0
            
            # Debt to Equity
            try:
                total_debt = bs.loc['Total Debt'][0] if 'Total Debt' in bs.index else (bs.loc['Long Term Debt'][0] if 'Long Term Debt' in bs.index else None)
                total_equity = bs.loc['Total Stockholder Equity'][0] if 'Total Stockholder Equity' in bs.index else (bs.loc['Stockholders Equity'][0] if 'Stockholders Equity' in bs.index else None)
                
                if total_debt and total_equity and total_equity > 0:
                    ratios['debt_to_equity'] = float(total_debt / total_equity)
                else:
                    ratios['debt_to_equity'] = 0.0
            except Exception as e:
                ratios['debt_to_equity'] = 0.0
            
            # Interest Coverage
            try:
                ebit = fin.loc['EBIT'][0] if 'EBIT' in fin.index else (fin.loc['Operating Income'][0] if 'Operating Income' in fin.index else None)
                interest_expense = fin.loc['Interest Expense'][0] if 'Interest Expense' in fin.index else None
                
                if ebit and interest_expense and interest_expense != 0:
                    ratios['interest_coverage'] = float(ebit / interest_expense)
                else:
                    ratios['interest_coverage'] = 0.0
            except Exception as e:
                ratios['interest_coverage'] = 0.0
            
            # OCF to Debt
            try:
                operating_cash_flow = cf.loc['Total Cash From Operating Activities'][0] if 'Total Cash From Operating Activities' in cf.index else None
                total_debt = bs.loc['Total Debt'][0] if 'Total Debt' in bs.index else (bs.loc['Long Term Debt'][0] if 'Long Term Debt' in bs.index else None)
                
                if operating_cash_flow and total_debt and total_debt != 0:
                    ratios['ocf_to_debt'] = float(operating_cash_flow / total_debt)
                else:
                    ratios['ocf_to_debt'] = 0.0
            except Exception as e:
                ratios['ocf_to_debt'] = 0.0
            
            # Receivables Turnover
            try:
                total_revenue = fin.loc['Total Revenue'][0] if 'Total Revenue' in fin.index else (fin.loc['Revenue'][0] if 'Revenue' in fin.index else None)
                accounts_receivable = bs.loc['Accounts Receivable'][0] if 'Accounts Receivable' in bs.index else (bs.loc['Net Receivables'][0] if 'Net Receivables' in bs.index else None)
                
                if total_revenue and accounts_receivable and accounts_receivable != 0:
                    ratios['receivables_turnover'] = float(total_revenue / accounts_receivable)
                else:
                    ratios['receivables_turnover'] = 0.0
            except Exception as e:
                ratios['receivables_turnover'] = 0.0
            
            # Payables Turnover Days
            try:
                accounts_payable = bs.loc['Accounts Payable'][0] if 'Accounts Payable' in bs.index else None
                cost_of_revenue = fin.loc['Cost Of Revenue'][0] if 'Cost Of Revenue' in fin.index else (fin.loc['Cost of Revenue'][0] if 'Cost of Revenue' in fin.index else None)
                
                if accounts_payable and cost_of_revenue and cost_of_revenue != 0:
                    ratios['payables_turnover_days'] = float((accounts_payable / cost_of_revenue) * 365)
                else:
                    ratios['payables_turnover_days'] = 60.0
            except Exception as e:
                ratios['payables_turnover_days'] = 60.0
            
            # Create DataFrame from ratios dictionary
            input_df = pd.DataFrame([ratios])
            
            # Debugging: Check what's in the input_df
            st.subheader("Input Ratios Used:")
            st.dataframe(input_df)
            
            # Check for too many NaNs
            if input_df.isnull().mean().mean() > 0.5:
                st.warning("⚠️ Insufficient financial data to predict bankruptcy risk for this company.")
            else:
                # Fill any remaining NaNs with zeros
                input_df = input_df.fillna(0)
                
                # Ensure all columns match the expected order
                expected_cols = ['working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
                                'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days']
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)
                
                # Scale the input data
                scaled = scaler.transform(input_df)
                
                st.write("Data scaled successfully!")
                
                # Use the model to predict
                try:
                    # Get prediction probabilities for both classes (0 and 1)
                    probabilities = model.predict_proba(scaled)
                    # Display full probabilities array to debug
                    st.write("Debug - Full probabilities:", probabilities)
                    
                    # Make sure we're getting the right probability (for class 1 - bankruptcy)
                    if probabilities.shape[1] >= 2:
                        pred = probabilities[0][1]  # Get probability for class 1
                    else:
                        pred = probabilities[0][0]  # Fallback if only one probability
                    
                    # Format the probability as percentage
                    st.write(f"Raw prediction probability: {pred:.4%}")
                    
                    pred_percent = pred * 100
                    
                    # Plot Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_percent,
                        number={'valueformat': '.2f'},
                        title={'text': "Bankruptcy Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#c95c5d"},  # Direct color code instead of variable
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    # Debug information
                    st.write("Debug - Model type:", type(model))
                    st.write("Debug - Scaled data shape:", scaled.shape)
                    # Try simple prediction
                    if model is not None:
                        simple_pred = model.predict(scaled)
                        st.write("Debug - Simple prediction:", simple_pred)
                    
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

    if st.button("Predict Bankruptcy Risk") and model is not None and scaler is not None:
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
                    'bar': {'color': red},
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
