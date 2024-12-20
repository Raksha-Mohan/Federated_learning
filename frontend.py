import streamlit as st
import pandas as pd
import requests
import numpy as np

def main():
    st.set_page_config(
        page_title="Patient Visit Predictor",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Patient Visit Frequency Predictor")

    # Create input fields
    st.header("Enter Patient Information")

    with st.form("prediction_form"):
        # Create columns for better layout
        col1, col2 = st.columns(2)

        # Clinical Measurements Section
        with col1:
            st.subheader("Clinical Measurements")
            gfr = st.number_input(
                "GFR (Glomerular Filtration Rate)",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                help="Normal range: 60-120 mL/min/1.73m²"
            )

            creatinine = st.number_input(
                "Serum Creatinine",
                min_value=0.0,
                max_value=15.0,
                value=1.0,
                help="Normal range: 0.7-1.3 mg/dL"
            )

            systolic_bp = st.number_input(
                "Systolic Blood Pressure",
                min_value=70,
                max_value=250,
                value=120,
                help="Normal range: 90-120 mmHg"
            )

        # Risk Factors Section
        with col2:
            st.subheader("Patient Risk Factors")
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=50,
                help="Patient's age in years"
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=50.0,
                value=25.0,
                help="Normal range: 18.5-24.9"
            )

            diabetes = st.selectbox(
                "Family History of Diabetes",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Family history of diabetes"
            )

        # Additional Information in Expandable Section
        with st.expander("Additional Clinical Information"):
            hba1c = st.number_input(
                "HbA1c",
                min_value=0.0,
                max_value=20.0,
                value=5.7,
                help="Normal range: 4.0-5.6%"
            )

            cholesterol = st.number_input(
                "Total Cholesterol",
                min_value=0.0,
                max_value=500.0,
                value=200.0,
                help="Normal range: <200 mg/dL"
            )

            medication_adherence = st.slider(
                "Medication Adherence",
                min_value=0,
                max_value=10,
                value=8,
                help="Scale of 0-10, where 10 is perfect adherence"
            )

        submit_button = st.form_submit_button("🔮 Predict Recommended Visits")

        if submit_button:
            try:
                features = [
                    age, bmi, systolic_bp, creatinine, gfr,
                    diabetes, hba1c, cholesterol, medication_adherence
                ]

                with st.spinner('Calculating recommendation...'):
                    input_data = {
                        "features": features
                    }

                    response = requests.post(
                        "http://localhost:8001/api/predict",
                        json=input_data,
                        timeout=10
                    )

                    if response.status_code == 200:
                        prediction = response.json()["prediction"]

                        st.success("✅ Prediction Complete")

                        res_col1, res_col2 = st.columns(2)

                        with res_col1:
                            st.metric(
                                label="Recommended Visits Per Month",
                                value=prediction
                            )

                        with res_col2:
                            if prediction >= 4:
                                st.warning("⚠️ High-frequency monitoring recommended")
                            elif prediction >= 2:
                                st.info("ℹ️ Regular monitoring recommended")
                            else:
                                st.success("✅ Routine monitoring recommended")
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend server. Please ensure it's running.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

    # Sidebar with information
    st.sidebar.header("ℹ️ About This Predictor")
    st.sidebar.markdown("""
    This tool predicts recommended monthly visit frequency based on:
    - Clinical measurements (GFR, Creatinine, BP)
    - Patient risk factors (Age, BMI, Family History)
    - Additional clinical information

    ### Normal Ranges:
    - GFR: 60-120 mL/min/1.73m²
    - Creatinine: 0.7-1.3 mg/dL
    - Systolic BP: 90-120 mmHg
    - BMI: 18.5-24.9
    - HbA1c: 4.0-5.6%
    """)

if __name__ == "__main__":
    main()