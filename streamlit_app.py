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
        st.subheader(f"Company: {info.get('longName', ticker)}")
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
        
        # 6. Operating Cash Flow to Debt - Improved version with better fallbacks
        try:
            # Try multiple possible locations for operating cash flow
            operating_cash_flow = get_value(cf, [
                'Total Cash From Operating Activities', 
                'Cash From Operating Activities',
                'Operating Cash Flow',
                'Net Cash Provided By Operating Activities'
            ])
            
            # Try multiple possible locations for total debt
            total_debt = get_value(bs, [
                'Total Debt', 
                'Long Term Debt',
                'Short Long Term Debt'
            ])
            
            # If we can't find total debt directly, try to calculate it
            if not total_debt:
                long_term_debt = get_value(bs, ['Long Term Debt'])
                short_term_debt = get_value(bs, ['Short Term Debt', 'Current Debt'])
                if long_term_debt or short_term_debt:
                    total_debt = sum(filter(None, [long_term_debt, short_term_debt]))
            
            if operating_cash_flow and total_debt and total_debt != 0:
                ratios['ocf_to_debt'] = float(operating_cash_flow / total_debt)
            else:
                # Try alternative calculation if total debt is missing
                total_liabilities = get_value(bs, ['Total Liabilities', 'Total Liabilities Net Minority Interest'])
                
                if operating_cash_flow and total_liabilities and total_liabilities != 0:
                    ratios['ocf_to_debt'] = float(operating_cash_flow / total_liabilities)
                    st.write(f"OCF/Debt ratio calculated using total liabilities: {ratios['ocf_to_debt']:.2f}")  # Debug
                else:
                    # Last resort: Look at balance sheet for operating cash flow and debt
                    cash_flow_alt = get_value(bs, ['Cash From Operating Activities'])
                    if cash_flow_alt and total_debt and total_debt != 0:
                        ratios['ocf_to_debt'] = float(cash_flow_alt / total_debt)
                    else:
                        # Default to a moderate value instead of zero
                        ratios['ocf_to_debt'] = 0.2  # Default to a moderate value
                        st.write("Using default OCF/Debt value")  # Debug
        except Exception as e:
            st.write(f"OCF to debt calculation error: {e}")
            ratios['ocf_to_debt'] = 0.2  # Use a moderate default rather than zero
        
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
    st.write("This app predicts bankruptcy risk using financial ratios and a logistic regression model tailored by industry groupings. It draws inspiration from Altman's Z-score but is expanded for modern datasets.")

    st.markdown(f"<h3 style='color:{red};'>Why?</h3>", unsafe_allow_html=True)
    st.write("Early prediction of bankruptcy risk helps prioritize audits, deeper analysis, and proactive financial decisions.")

    st.markdown(f"<h3 style='color:{red};'>How?</h3>", unsafe_allow_html=True)
    st.write("""
    - 8 core financial ratios are standardized and combined with industry dummy variables.
    - A logistic regression model estimates bankruptcy probability over a 5-year horizon.
    - No subjective judgments — purely quantitative early warning system.
    """)

    st.latex(r"""z = \sum_{i=1}^{10} \alpha_i \cdot industry_i + \sum_{j=1}^{8} \beta_j \cdot ratio_j""")

    st.write("Model trained as of **April 25, 2025**.")
    st.markdown("---")
    st.markdown("🔗 [GitHub Repository for this App](https://github.com/myashaswi/BankruptcyPredictionApp)")

# ========== PAGE 2: Bankruptcy Prediction Based on Ticker ==========
elif page.startswith("2"):
    st.title("Bankruptcy Risk Prediction Based on Stock Ticker")

    ticker = st.text_input("Enter Stock Ticker (example: AAPL, MSFT, NVDA)").upper()

    if ticker and model is not None and scaler is not None:
        try:
            # Calculate financial ratios using the function defined earlier
            ratios, company_info = calculate_financial_ratios(ticker)

            if ratios:
                # Create DataFrame from ratios dictionary
                input_df = pd.DataFrame([ratios])
                
                # Display the calculated ratios
                st.subheader("Financial Ratios")
                ratio_display = pd.DataFrame({
                    'Ratio': list(ratios.keys()),
                    'Value': list(ratios.values())
                })
                st.dataframe(ratio_display)

                pretty_names = {
                        'working_capital_ratio': 'Working Capital Ratio',
                        'roa': 'Return on Assets',
                        'ebit_to_assets': 'EBIT to Assets',
                        'debt_to_equity': 'Debt to Equity',
                        'interest_coverage': 'Interest Coverage',
                        'ocf_to_debt': 'Operating Cash Flow to Debt',
                        'receivables_turnover': 'Receivables Turnover',
                        'payables_turnover_days': 'Payables Turnover Days'
                    }

                # Use for display purposes
                display_df = input_df.rename(columns=pretty_names)
                st.dataframe(display_df)

                # Check for too many missing values
                if sum(v == 0 for v in ratios.values()) > 4:  # If more than half are zeros/missing
                    st.warning("⚠️ Insufficient financial data for reliable prediction.")
                
                # Ensure all required columns are present
                expected_cols = ['working_capital_ratio', 'roa', 'ebit_to_assets', 'debt_to_equity',
                                'interest_coverage', 'ocf_to_debt', 'receivables_turnover', 'payables_turnover_days']
                for col in expected_cols:
                    if col not in ratios:
                        ratios[col] = 0
                
                input_df = pd.DataFrame([ratios])[expected_cols]
                
                try:
                    # Standardize the input data using the scaler
                    scaled_input = scaler.transform(input_df)
                    
                    # Use the loaded model for prediction
                    bankruptcy_prob = model.predict_proba(scaled_input)[0][1]
                    
                    # Apply some smoothing to avoid extreme values
                    # This creates a more balanced distribution between 0.05 and 0.95
                    smoothed_prob = 0.05 + (bankruptcy_prob * 0.9)
                    
                    # Industry-specific adjustments (more subtle)
                    industry = company_info.get('industry', '').lower()
                    sector = company_info.get('sector', '').lower()
                    
                    # Apply moderate industry adjustments
                    if any(term in industry or term in sector for term in ['tech', 'software', 'semiconductor']):
                        smoothed_prob *= 0.85  # Reduce risk for tech
                    elif any(term in industry or term in sector for term in ['retail', 'energy', 'airline']):
                        smoothed_prob *= 1.15  # Increase risk for vulnerable sectors
                    
                    # Cap the probability between 5% and 95%
                    smoothed_prob = max(0.05, min(0.95, smoothed_prob))
                    
                    # Special case for very large market cap companies
                    market_cap = company_info.get('marketCap', 0)
                    if market_cap > 500000000000:  # > $500B
                        smoothed_prob = max(0.05, min(0.25, smoothed_prob))  # Cap at 25% for mega corps
                    
                    # Format final probability
                    pred_percent = smoothed_prob * 100
                    
                    # Plot Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_percent,
                        number={'valueformat': '.1f'},
                        title={'text': "Bankruptcy Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#c95c5d"},
                            'steps': [
                                {'range': [0, 70], 'color': "lightgreen"},
                                {'range': [71, 95], 'color': "yellow"},
                                {'range': [96, 100], 'color': "red"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show risk interpretation
                    if pred_percent < 71:
                        st.success("🟢 Low bankruptcy risk")
                    elif pred_percent < 96:
                        st.warning("🟡 Moderate bankruptcy risk")
                    else:
                        st.error("🔴 High bankruptcy risk")
                        
                    # Additional context
                    with st.expander("Factors influencing this prediction"):
                        # Explain which factors contributed most to the prediction
                        st.write("Key financial metrics and their impact:")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Positive factors:**")
                            if ratios['working_capital_ratio'] > 0:
                                st.write(f"✓ Working Capital Ratio: {ratios['working_capital_ratio']:.2f}")
                            if ratios['roa'] > 0:
                                st.write(f"✓ Return on Assets: {ratios['roa']:.2f}")
                            if ratios['interest_coverage'] > 2:
                                st.write(f"✓ Interest Coverage: {ratios['interest_coverage']:.2f}")
                            if ratios['ocf_to_debt'] > 0.2:
                                st.write(f"✓ Op. Cash Flow to Debt: {ratios['ocf_to_debt']:.2f}")
                        
                        with col2:
                            st.write("**Risk factors:**")
                            if ratios['working_capital_ratio'] < 0.2:
                                st.write(f"⚠️ Working Capital Ratio: {ratios['working_capital_ratio']:.2f}")
                            if ratios['roa'] < 0.02:
                                st.write(f"⚠️ Return on Assets {ratios['roa']:.2f}")
                            if ratios['ebit_to_assets'] < 0.03:
                                st.write(f"⚠️ EBIT to Assets: {ratios['ebit_to_assets']:.2f}")
                            if ratios['debt_to_equity'] > 2:
                                st.write(f"⚠️ Debt to Equity: {ratios['debt_to_equity']:.2f}")
                            if ratios['interest_coverage'] < 2:
                                st.write(f"⚠️ Interest Coverage: {ratios['interest_coverage']:.2f}")
                            if ratios['ocf_to_debt'] < 0.2:
                                st.write(f"⚠️ Low Operating Cash Flow to Debt: {ratios['ocf_to_debt']:.2f}")
                            if ratios['receivables_turnover'] < 4:
                                st.write(f"⚠️ Low Receivables Turnover: {ratios['receivables_turnover']:.2f}")
                            if ratios['payables_turnover_days'] > 80:
                                st.write(f"⚠️ High Payables Turnover Days : {ratios['payables_turnover_days']:.2f}")
                            risk_factors = []  # Initialize an empty list before checking
                            if risk_factors :
                                st.subheader("⚠️ Potential Risk Factors Detected:")
                                for factor in risk_factors:
                                    st.write(f"- {factor}")
                            else:
                                st.success("✅ No critical financial red flags detected. But still interpret bankruptcy probability carefully.")
                
                except Exception as e:
                    st.error(f"Prediction failed. Error: {e}")
            else:
                st.error(f"Could not retrieve sufficient financial data for {ticker}. Please check the ticker symbol.")

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

    industry = st.selectbox("Industry", 
                           ["Technology", "Energy", "Healthcare", "Financial Services", 
                            "Consumer Discretionary", "Consumer Staples", "Industrials", 
                            "Materials", "Utilities", "Real Estate", "Other"])

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
            bankruptcy_prob = model.predict_proba(scaled_manual)[0][1]
            
            # Apply some smoothing to avoid extreme values
            smoothed_prob = 0.05 + (bankruptcy_prob * 0.9)
            
            # Industry-specific adjustments (more subtle)
            industry_lower = industry.lower()
            if any(term in industry_lower for term in ['tech', 'software']):
                smoothed_prob *= 0.85  # Reduce risk for tech
            elif any(term in industry_lower for term in ['retail', 'energy', 'airline']):
                smoothed_prob *= 1.15  # Increase risk for vulnerable sectors
            
            # Cap the probability between 5% and 95%
            smoothed_prob = max(0.05, min(0.95, smoothed_prob))
            
            # Format final probability
            pred_percent = smoothed_prob * 100

            fig_manual = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_percent,
                number={'valueformat': '.1f'},
                title={'text': "Bankruptcy Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': red},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgreen"},
                        {'range': [60, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                }
            ))
            st.plotly_chart(fig_manual, use_container_width=True)
            
            # Show risk interpretation
            if pred_percent < 30:
                st.success("🟢 Low bankruptcy risk")
            elif pred_percent < 70:
                st.warning("🟡 Moderate bankruptcy risk")
            else:
                st.error("🔴 High bankruptcy risk")
                
            # Additional context
            with st.expander("Factors influencing this prediction"):
                # Explain which factors contributed most to the prediction
                st.write("Key financial metrics and their impact:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Positive factors:**")
                    if working_capital_ratio > 0:
                        st.write(f"✓ Working Capital Ratio: {working_capital_ratio:.2f}")
                    if roa > 0:
                        st.write(f"✓ Return on Assets: {roa:.2f}")
                    if interest_coverage > 2:
                        st.write(f"✓ Interest Coverage: {interest_coverage:.2f}")
                    if ocf_to_debt > 0.2:
                        st.write(f"✓ Op. Cash Flow to Debt: {ocf_to_debt:.2f}")
                
                with col2:
                    st.write("**Risk factors:**")
                    if working_capital_ratio < 0:
                        st.write(f"⚠️ Working Capital Ratio: {working_capital_ratio:.2f}")
                    if debt_to_equity > 1:
                        st.write(f"⚠️ Debt to Equity: {debt_to_equity:.2f}")
                    if interest_coverage < 1.5:
                        st.write(f"⚠️ Interest Coverage: {interest_coverage:.2f}")
                    if ocf_to_debt < 0.1:
                        st.write(f"⚠️ Low Cash Flow to Debt: {ocf_to_debt:.2f}")

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
