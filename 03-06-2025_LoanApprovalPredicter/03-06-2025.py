import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F5F5F5; }
    .title { font-size: 36px; font-weight: bold; color: #4A90E2; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🏦 Loan Approval Predictor</p>', unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('loan-train.csv')
    df.drop(columns=['Loan_ID'], inplace=True)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    le_dict = {}
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict

df, le_dict = load_and_preprocess_data()
features = df.drop('Loan_Status', axis=1).columns

model = joblib.load("loan_model.pkl")

with st.sidebar:
    st.markdown("## 💡 Tips for Approval")
    st.markdown("- Higher **Credit History** (1.0) boosts chances.")
    st.markdown("- Lower **Loan Amount** requested improves odds.")
    st.markdown("- Stable **Income** is favorable.")
    st.markdown("- **Graduate** education gives advantage.")
    st.markdown("- **Semiurban/Urban** properties more likely to be approved.")

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
    gender_enc = 1 if gender == "Male" else 0
    married_enc = 1 if married == "Yes" else 0
    dependents_enc = 3 if dependents == "3+" else int(dependents)
    education_enc = 0 if education == "Graduate" else 1
    self_employed_enc = 1 if self_employed == "Yes" else 0
    credit_history_enc = 1.0 if credit_history.startswith("Has") else 0.0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area_enc = property_map[property_area]

    input_data = np.array([[gender_enc, married_enc, dependents_enc, education_enc, self_employed_enc,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history_enc, property_area_enc]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

st.subheader("📊 Feature Importance (Model Insights)")
importances = model.feature_importances_
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Most Influential Factors")
st.plotly_chart(fig, use_container_width=True)
