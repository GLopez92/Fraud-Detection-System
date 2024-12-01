# **Fraud Detection System**

### **Overview**
The Fraud Detection System is a machine learning-based project designed to identify fraudulent transactions in credit card datasets. Using techniques like data preprocessing, oversampling with SMOTE, and modeling with Logistic Regression and Random Forest, this project highlights the imbalance in fraud detection data and offers interpretable solutions using SHAP visualizations.

---

### **Goals**
- **Analyze Credit Card Transactions:** Process and explore the dataset to identify fraudulent transactions effectively.
- **Overcome Imbalanced Data Issues:** Use SMOTE to balance the dataset and ensure accurate predictions.
- **Compare Models:** Evaluate Logistic Regression and Random Forest classifiers for fraud detection.
- **Visualize Results:** Provide clear visual insights through SHAP values and feature importances.
- **Deploy Model:** Offer a Streamlit web app for easy user interaction.

---

### **Setup Instructions**

#### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.8 or higher
- Anaconda/Miniconda
- Git (for version control)

#### **2. Installation Steps**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fraud-detection
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the project:
   ```bash
   python fraud_detection.py
   ```

#### **3. Optional: Streamlit App**
To use the Streamlit app:
1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Run the app:
   ```bash
   streamlit run fraud_detection_app.py
   ```

---

### **Screenshots**

#### **Class Distribution**
Shows the significant imbalance between fraudulent and non-fraudulent transactions.

#### **Logistic Regression Evaluation**
Evaluates the performance of Logistic Regression with metrics like precision, recall, and AUC-ROC.

#### **Random Forest Feature Importances**
Highlights the most influential features in the dataset.

#### **SHAP Summary Plot**
Visualizes the impact of features on predictions.

---

### **Example Results**
- **Logistic Regression AUC-ROC:** `0.98`
- **Random Forest AUC-ROC:** `0.99`
- **SHAP Interpretation:** Provides insights into feature contributions.

---

### **Project Structure**
- **`fraud_detection.py:`** Main script for processing, modeling, and evaluation.
- **`fraud_detection_app.py:`** Streamlit app for user interaction.
- **`creditcard.csv:`** Dataset for fraud detection.
- **`README.md:`** Documentation.

---

### **Future Enhancements**
- Integrate additional machine learning models like XGBoost.
- Explore unsupervised learning for anomaly detection.
- Build an interactive dashboard for real-time fraud detection.

---
![Screenshot 2024-11-30 013803](https://github.com/user-attachments/assets/39083c44-6bfd-48e2-9a6c-201874b3f93b)


