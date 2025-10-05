import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="wide",
)

# --- LOAD MODELS AND ENCODERS ---
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders from pickle files."""
    try:
        with open("customer_churn_model.pkl", "rb") as f_model:
            model_data = pickle.load(f_model)
        with open("encoders.pkl", "rb") as f_encoders:
            encoders = pickle.load(f_encoders)
        return model_data, encoders
    except FileNotFoundError:
        st.error("Model or encoder files not found. Make sure 'customer_churn_model.pkl' and 'encoders.pkl' are in the same directory.")
        return None, None

model_data, encoders = load_model_and_encoders()

if model_data and encoders:
    model = model_data["model"]
    feature_names = model_data["features_names"]

# --- APP TITLE AND DESCRIPTION ---
st.title("Customer Churn Prediction System")
st.markdown("Enter the customer's details below to predict their likelihood of churning.")

# --- CREATE INPUT FORM ---
st.header("Customer Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

# Using index=None for selectbox and value=None for number_input creates placeholders
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select an option...")
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=None, placeholder="Select an option...")
    partner = st.selectbox("Partner", ["No", "Yes"], index=None, placeholder="Select an option...")
    dependents = st.selectbox("Dependents", ["No", "Yes"], index=None, placeholder="Select an option...")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=None, placeholder="e.g., 24")

with col2:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=None, placeholder="Select an option...")
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], index=None, placeholder="Select an option...")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=None, placeholder="Select an option...")
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=None, placeholder="Select an option...")
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=None, placeholder="Select an option...")

st.divider()

col3, col4 = st.columns(2)

with col3:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=None, placeholder="Select an option...")
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=None, placeholder="Select an option...")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=None, placeholder="Select an option...")
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=None, placeholder="Select an option...")

with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=None, placeholder="Select a contract type...")
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=None, placeholder="Select an option...")
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=None, placeholder="Select a payment method...")
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=None, placeholder="e.g., 70.50")
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=None, placeholder="e.g., 1500.65")

# --- PREDICTION BUTTON ---
if st.button("Predict Churn", type="primary"):
    # Check if all fields are filled
    all_inputs = [
        gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
        internet_service, online_security, online_backup, device_protection, tech_support,
        streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
        monthly_charges, total_charges
    ]

    if None in all_inputs:
        st.warning("⚠️ Please fill in all the fields to get a prediction.")
    else:
        # Create a dictionary from the inputs
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        # Create a DataFrame from the dictionary
        input_df = pd.DataFrame([input_data])
        input_df_encoded = input_df.copy()

        # Apply the stored label encoders to the categorical columns
        for column, encoder in encoders.items():
            if column in input_df_encoded.columns:
                input_df_encoded[column] = encoder.transform(input_df_encoded[column])
        
        # Ensure the order of columns matches the training data
        input_df_encoded = input_df_encoded[feature_names]

        # --- MAKE PREDICTION ---
        prediction = model.predict(input_df_encoded)
        prediction_proba = model.predict_proba(input_df_encoded)

        churn_probability = prediction_proba[0][1] * 100

        # --- DISPLAY RESULT ---
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**This customer is LIKELY to churn.** (Churn Probability: {churn_probability:.2f}%)")
        else:
            st.success(f"**This customer is UNLIKELY to churn.** (Churn Probability: {churn_probability:.2f}%)")