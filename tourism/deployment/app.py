import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="jackson-daniel/tourism", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a customer opting to buy a tourism package.
Please enter the requested data below to get a prediction.
""")

# User input widgets
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=1, value=15, step=1)
occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
no_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, value=2, step=1)
no_of_followups = st.number_input("Number Of Followups", min_value=0, value=1, step=1)
product_pitched  = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Standard"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number Of Trips (annually)", min_value=0, value=3, step=1)
passport = st.checkbox("Has Passport?") # Converts to 0 or 1 implicitly
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.checkbox("Owns Car?") # Converts to 0 or 1 implicitly
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, value=25000, step=500)


if st.button("Predict Purchase"):
    # Prepare raw input data for transformation
    input_data =  pd.DataFrame([{
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': float(duration_of_pitch),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': no_of_person_visiting,
        'NumberOfFollowups': float(no_of_followups),
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': float(preferred_property_star),
        'MaritalStatus': marital_status,
        'NumberOfTrips': float(number_of_trips),
        'Passport': 1 if passport else 0,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': 1 if own_car else 0,
        'NumberOfChildrenVisiting': float(number_of_children_visiting),
        'Designation': designation,
        'MonthlyIncome': float(monthly_income)
    }])

 

    prediction = model.predict(input_data)[0]
    result = "will purchase" if prediction == 1 else "will not purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts the customer **{result}** a tourism package.") 
