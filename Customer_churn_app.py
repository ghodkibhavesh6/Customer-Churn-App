import streamlit as st
import pandas as pd
import pickle

# Load saved model 
model = pickle.load(open("churn_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("📞 Customer Churn Prediction System")

st.subheader("Enter Customer Details")

# input from user
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (Months)", 0, 100, 12)
usage = st.number_input("Usage Frequency", 0, 100, 10)
support_calls = st.number_input("Support Calls", 0, 20, 1)
payment_delay = st.number_input("Payment Delay (Days)", 0, 60, 5)
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend", 0, 10000, 500)
last_interaction = st.number_input("Last Interaction (Days Ago)", 0, 60, 10)

# make input dataframe
input_data = {
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Subscription Type_Premium": 1 if subscription == "Premium" else 0,
    "Subscription Type_Standard": 1 if subscription == "Standard" else 0,
    "Contract Length_Monthly": 1 if contract == "Monthly" else 0,
    "Contract Length_Quarterly": 1 if contract == "Quarterly" else 0,
}

input_df = pd.DataFrame([input_data])

# Match training columns
input_df = input_df.reindex(columns=model_columns, fill_value=0)

#predict the output
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
