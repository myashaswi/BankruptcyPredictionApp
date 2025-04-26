import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load the saved model
model = joblib.load('model.pkl')

# Industry mapping
industry_mapping = {
    3520: "Pharmaceuticals, Biotechnology & Life Sciences",
    3510: "Health Care Equipment & Services",
    1010: "Energy",
    2550: "Consumer Discretionary Distribution & Retail",
    4510: "Software & Services",
    2010: "Capital Goods",
    2530: "Consumer Services",
    3030: "Household & Personal Products",
    1510: "Materials",
    4520: "Technology Hardware & Equipment"
}

# Expected feature columns for model prediction
expected_features = [
    'working_capital_ratio',
    'roa',
    'ebit_to_assets',
    'debt_to_equity',
    'interest_coverage',
    'ocf_to_debt',
    'receivables_turnover',
    'payables_turnover_days'
]

# Streamlit page setup
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")
st.sidebar.title("📊 Bankruptcy Prediction App")
st.sidebar.markdown("""
Upload your company financial ratios dataset (must include **ggroup** code and 8 financial features).

App will predict bankruptcy risk and summarize by Industry!
""")

st.title("🚀 Bankruptcy Prediction App")

uploaded_file = st.file_uploader("Upload your company financial ratios CSV", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    # Validate columns
    required_columns = ['ggroup'] + expected_features
    if all(col in input_data.columns for col in required_columns):
        st.success("✅ File uploaded successfully!")
        
        # Map ggroup codes to industry names
        input_data['Industry'] = input_data['ggroup'].map(industry_mapping)
        
        # Drop rows without a valid industry mapping
        input_data = input_data.dropna(subset=['Industry'])
        
        # Show the uploaded data
        st.subheader("Uploaded Data Preview:")
        st.dataframe(input_data[['Industry'] + expected_features].head(10))
        
        # Make predictions
        X = input_data[expected_features]
        predictions = model.predict(X)
        prediction_probs = model.predict_proba(X)[:, 1]  # Probability of bankruptcy
        
        input_data['Prediction'] = predictions
        input_data['Bankruptcy Probability'] = prediction_probs
        
        # Summary metrics
        total_companies = len(input_data)
        total_bankruptcies = (input_data['Prediction'] == 1).sum()
        bankruptcy_percentage = (total_bankruptcies / total_companies) * 100
        
        st.metric(label="Total Companies Analyzed", value=f"{total_companies}")
        st.metric(label="Companies Likely to go Bankrupt", value=f"{total_bankruptcies} ({bankruptcy_percentage:.2f}%)")
        
        # Group by industry
        industry_summary = input_data.groupby('Industry').agg(
            Total_Companies=('Prediction', 'count'),
            Bankruptcies=('Prediction', 'sum')
        ).reset_index()
        industry_summary['Bankruptcy_Rate (%)'] = (industry_summary['Bankruptcies'] / industry_summary['Total_Companies']) * 100
        
        st.subheader("📊 Bankruptcy Risk by Industry")
        st.dataframe(industry_summary.style.format({"Bankruptcy_Rate (%)": "{:.2f}"}))
        
        # Bar chart visualization
        chart = alt.Chart(industry_summary).mark_bar().encode(
            x=alt.X('Industry', sort='-y', title="Industry"),
            y=alt.Y('Bankruptcies', title='Number of Bankruptcies'),
            color=alt.Color('Bankruptcies', scale=alt.Scale(scheme='oranges'))
        ).properties(
            width=800,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("Detailed Company-Level Predictions")
        st.dataframe(input_data[['Industry'] + expected_features + ['Prediction', 'Bankruptcy Probability']].style.format({"Bankruptcy Probability": "{:.2f}"}))

    else:
        st.error(f"❌ Uploaded file must contain the following columns: {required_columns}")
else:
    st.info("👈 Please upload a CSV file to get started.")
