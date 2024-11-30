# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 02:14:43 2024

@author: GuillermoLopez
"""

# fraud_detection_app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Title of the app
st.title("Fraud Detection System")

# Upload the model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type="csv")

if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    # Check if the required columns exist
    required_columns = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", 
                        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", 
                        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", 
                        "V28", "Amount"]

    if all(col in data.columns for col in required_columns):
        # Make predictions
        predictions = model.predict(data[required_columns])

        # Add predictions to the data
        data["Fraud Prediction"] = predictions

        # Display results
        st.write("Prediction Results:")
        st.dataframe(data[["Fraud Prediction"]])

        # Downloadable version of results
        csv = data.to_csv(index=False)
        st.download_button(label="Download Predictions as CSV", 
                           data=csv, 
                           file_name="fraud_predictions.csv", 
                           mime="text/csv")
    else:
        st.error(f"The uploaded file is missing required columns: {', '.join(required_columns)}")
else:
    st.info("Please upload a CSV file containing transaction data.")
