# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:19:18 2024

@author: GuillermoLopez
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import shap
import streamlit as st

# Load the dataset
data_path = "C:/Users/willw/fraud_detection/creditcard.csv"
data = pd.read_csv(data_path)

# Handle missing values (if any)
data.fillna(data.median(), inplace=True)

# Visualize the class distribution
sns.countplot(data['Class'])
plt.title("Fraud vs Non-Fraud Distribution")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# Feature scaling for 'Amount'
data['Amount'] = StandardScaler().fit_transform(data[['Amount']])

# Handle imbalanced dataset using SMOTE
X = data.drop(columns=['Class'])
y = data['Class']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_y_pred = log_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_y_pred))
print("Logistic Regression AUC-ROC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))
print("Random Forest AUC-ROC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Visualization of Feature Importance (Random Forest)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importances - Random Forest")
plt.show()

# SHAP interpretation for Logistic Regression
explainer = shap.Explainer(log_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Save the logistic regression model
joblib.dump(log_model, 'fraud_model.pkl')

# Streamlit app for deployment
st.title("Fraud Detection System")

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type="csv")

if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    # Feature scaling for 'Amount'
    if "Amount" in data.columns:
        data['Amount'] = StandardScaler().fit_transform(data[['Amount']])

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







