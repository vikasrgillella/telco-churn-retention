from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Telco Churn Scoring", layout="centered")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "reports" / "churn_model_logreg.joblib"

st.title("Telco Customer Churn Scoring")
st.write("Scores a new customer using the trained pipeline from the notebook.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run the notebook first.")
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("form"):
    tenure = st.number_input("tenure", 0, 120, 12)
    monthly = st.number_input("MonthlyCharges", 0.0, 300.0, 70.0)
    total = st.number_input("TotalCharges", 0.0, 20000.0, 800.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
    ok = st.form_submit_button("Score")

if ok:
    X_new = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
    }])
    prob = model.predict_proba(X_new)[:, 1][0]
    st.metric("Churn Probability", f"{prob:.2%}")
