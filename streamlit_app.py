import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('model.pkl')

st.title("Bankruptcy Prediction App 🚀")
st.write("""
Upload a CSV file containing company financial ratios to predict bankruptcy risk. 
If you don't upload anything, we will use a sample company data.
""")

# Define expected CSV columns
expected_columns = [
    'working_capital_ratio',
    'roe',
    'roa',
    'gross_profit_margin',
    'interest_coverage',
    'debt_to_equity_ratio',
    'current_ratio',
    'asset_turnover'
]

# Sample data if no file uploaded
sample_data = pd.DataFrame({
    'working_capital_ratio': [0.5],
    'roe': [0.1],
    'roa': [0.08],
    'gross_profit_margin': [0.3],
    'interest_coverage': [2.0],
    'debt_to_equity_ratio': [1.2],
    'current_ratio': [1.5],
    'asset_turnover': [0.7],
})

uploaded_file = st.file_uploader("Upload your company financial ratios CSV file", type="csv")

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)

        # Check if required columns exist
        if all(col in input_data.columns for col in expected_columns):
            st.success("✅ CSV uploaded successfully and looks good!")
            st.dataframe(input_data)
        else:
            st.error(f"❌ CSV missing required columns. Please make sure it has these columns: {expected_columns}")
            st.stop()

    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()
else:
    st.info("No CSV uploaded. Using sample data instead.")
    input_data = sample_data
    st.dataframe(input_data)

# Predict
st.subheader("Prediction Results:")
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

for idx, (pred, prob) in enumerate(zip(prediction, prediction_proba)):
    result = "❌ Likely to go bankrupt" if pred == 1 else "✅ Not likely to go bankrupt"
    st.write(f"Company {idx+1}: {result} (Probability of bankruptcy: {prob[1]:.2f})")
