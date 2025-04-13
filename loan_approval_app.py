import streamlit as st
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open('loan_model.pkl', 'rb'))
le_status = pickle.load(open('status_encoder.pkl', 'rb'))
le_edu = pickle.load(open('edu_encoder.pkl', 'rb'))
le_emp = pickle.load(open('emp_encoder.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")

st.markdown("### Enter applicant details to check loan eligibility:")

# Input fields
dependents = st.selectbox("Number of Dependents", options=list(range(10)))
education = st.selectbox("Education", options=le_edu.classes_)
self_employed = st.selectbox("Self Employed", options=le_emp.classes_)
income = st.number_input("Annual Income (₹)", value=500000, step=10000)
loan_amount = st.number_input("Loan Amount (₹)", value=200000, step=10000)
loan_term = st.number_input("Loan Term (months)", value=12, step=1)
cibil = st.slider("CIBIL Score", min_value=300, max_value=900, value=750)
res_asset = st.number_input("Residential Asset Value (₹)", value=0, step=10000)
comm_asset = st.number_input("Commercial Asset Value (₹)", value=0, step=10000)
lux_asset = st.number_input("Luxury Asset Value (₹)", value=0, step=10000)
bank_asset = st.number_input("Bank Asset Value (₹)", value=0, step=10000)

# Encode categorical features using label encoders
edu_encoded = le_edu.transform([education])[0]
emp_encoded = le_emp.transform([self_employed])[0]

# Prepare input
input_data = np.array([
    dependents,
    edu_encoded,
    emp_encoded,
    income / 1e6,
    loan_amount / 1e6,
    loan_term,
    cibil,
    res_asset / 1e6,
    comm_asset / 1e6,
    lux_asset / 1e6,
    bank_asset / 1e6
]).reshape(1, -1)

# Prediction button
if st.button("🔍 Predict Loan Approval"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    result = le_status.inverse_transform([pred])[0]

    if result == "Approved":
        st.success("✅ Loan is likely to be Approved!")
    else:
        st.error("❌ Loan will likely be Rejected.")

    # Show probability
    st.markdown("### 🔢 Prediction Confidence:")
    st.info(f"✅ Approved: {proba[0]*100:.2f}%\n❌ Rejected: {proba[1]*100:.2f}%")

    # Warnings
    if loan_term <= 1:
        st.warning("⚠️ Very short loan term may reduce approval chances.")
    if sum([res_asset, comm_asset, lux_asset, bank_asset]) == 0:
        st.warning("⚠️ No assets provided. This may hurt approval likelihood.")

    # Debug info
    st.markdown("### 🛠️ Debug Info")
    st.code(f"Raw model prediction: {pred}\nInput data: {input_data.tolist()}")

# Test case
st.markdown("---")
st.subheader("🧪 Test with Verified Approved Example")

if st.button("Run Sample Approved Case"):
    test_case = np.array([
        0,
        le_edu.transform(['Graduate'])[0],
        le_emp.transform(['Yes'])[0],
        800000 / 1e6,
        2200000 / 1e6,
        20,
        782,
        1300000 / 1e6,
        800000 / 1e6,
        2800000 / 1e6,
        600000 / 1e6
    ]).reshape(1, -1)

    test_pred = model.predict(test_case)[0]
    test_proba = model.predict_proba(test_case)[0]
    test_result = le_status.inverse_transform([test_pred])[0]

    if test_result == "Approved":
        st.success("✅ Model correctly predicts this approved case!")
    else:
        st.error("❌ Model misclassified this known approved case.")

    st.markdown(f"**Confidence:**")
    st.info(f"✅ Approved: {test_proba[0]*100:.2f}%\n❌ Rejected: {test_proba[1]*100:.2f}%")
