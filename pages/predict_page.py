import streamlit as st
import pandas as pd
import joblib
import os

def predict_page():
    st.title("Predict Attrition")

    # Load the model, encoder mapping, and scaler
    model = joblib.load(os.path.join('models', 'model.pkl'))
    encoder_mapping = joblib.load(os.path.join('models', 'encoder_mapping.pkl'))
    scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

    # Create a form to collect input data
    with st.form("prediction_form"):
        # Create three columns for input
        col1, col2, col3 = st.columns(3)

        # Column 1: Personal Info
        with col1:
            st.header("Personal Info")
            senior_citizen = st.selectbox("Are you a senior citizen?", ["No", "Yes"])
            partner = st.selectbox("Do you have a partner?", ["No", "Yes"])
            dependents = st.selectbox("Do you have dependents?", ["No", "Yes"])
            tenure = st.number_input("Enter your tenure (in months)", min_value=0)

        # Column 2: Service Info
        with col2:
            st.header("Service Info")
            internet_service = st.selectbox("Select your internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Do you have online security?", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Do you have online backup?", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Do you have device protection?", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Do you have tech support?", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Do you have streaming TV?", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Do you have streaming movies?", ["No", "Yes", "No internet service"])

        # Column 3: Billing Info
        with col3:
            st.header("Billing Info")
            contract = st.selectbox("Select your contract type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Do you have paperless billing?", ["No", "Yes"])
            payment_method = st.selectbox("Select your payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Enter your monthly charges", min_value=0.0)
            total_charges = st.number_input("Enter your total charges", min_value=0.0)

        # Submit button for the form
        submitted = st.form_submit_button("Predict")

        # Placeholder for prediction
        if submitted:
            # Collect input data
            input_data = {
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
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

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables using the encoder mapping
            for feature in encoder_mapping:
                input_df[feature] = input_df[feature].map(encoder_mapping[feature])

            # Scale numerical features
            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])

            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display output
            st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
            st.write(f"Prediction Probability: {prediction_proba[0]}")