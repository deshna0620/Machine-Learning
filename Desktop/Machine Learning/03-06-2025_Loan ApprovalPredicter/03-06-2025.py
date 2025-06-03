import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
import plotly.express as px
import os

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F5F5F5; }
    .title { font-size: 36px; font-weight: bold; color: #4A90E2; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🏦 Loan Approval Predictor</p>', unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_and_train():
    df = pd.read_csv('loan-train.csv')
    df.drop(columns=['Loan_ID'], inplace=True)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'loan_model.pkl')

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    importances = model.feature_importances_

    return report, cm, importances, X.columns

# Train if not saved
if not os.path.exists("loan_model.pkl"):
    report, cm, importances, features = load_and_train()
    st.success("✅ Model trained and saved successfully!")
else:
    model = joblib.load("loan_model.pkl")
    df = pd.read_csv("loan-train.csv")
    df.drop(columns=['Loan_ID'], inplace=True)
    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    model.fit(X, y)
    importances = model.feature_importances_
    features = X.columns

# Sidebar tips
with st.sidebar:
    st.markdown("## 💡 Tips for Approval")
    st.markdown("- Higher **Credit History** (1.0) boosts chances.")
    st.markdown("- Lower **Loan Amount** requested improves odds.")
    st.markdown("- Stable **Income** is favorable.")
    st.markdown("- **Graduate** education gives advantage.")
    st.markdown("- **Semiurban/Urban** properties more likely to be approved.")

# Input fields
st.subheader("🔍 Enter Applicant Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

with col3:
    loan_term = st.selectbox("Loan Term (months)", [360.0, 180.0, 120.0, 300.0, 240.0, 60.0, 84.0, 36.0])
    credit_history = st.selectbox("Credit History", ["Has History (1)", "No History (0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Approval"):
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1.0 if credit_history.startswith("Has") else 0.0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_map[property_area]

    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, property_area]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

# Feature importance plot
st.subheader("📊 Feature Importance (Model Insights)")
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Most Influential Factors")
st.plotly_chart(fig, use_container_width=True)
