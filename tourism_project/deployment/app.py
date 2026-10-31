import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to **purchase the Wellness Tourism Package**
offered by *Visit with Us*, based on their profile and sales interaction details.
Enter the customer details below to get a prediction.
""")

# User input - customer details
st.subheader("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Unmarried", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=500000, value=23000, step=500)
number_of_trips = st.number_input("Average Number of Trips per Year", min_value=1, max_value=30, value=3)
passport = st.selectbox("Holds a Valid Passport?", ["Yes", "No"])
own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=3)
number_of_children_visiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=1)
preferred_property_star = st.selectbox("Preferred Hotel Rating (Stars)", [3, 4, 5])

# User input - sales interaction details
st.subheader("Sales Interaction Details")
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=150, value=15)
number_of_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=4)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)

# Assemble input into a DataFrame with the same columns used in training
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

st.subheader("Input Summary")
st.write(input_data)

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"This customer is **likely to purchase** the Wellness Tourism Package "
                   f"(probability: {probability:.1%}).")
    else:
        st.warning(f"This customer is **unlikely to purchase** the Wellness Tourism Package "
                   f"(probability: {probability:.1%}).")
