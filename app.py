import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("careermodel.joblib")
label_encoders = joblib.load("careerencoders.joblib")  # Ensure this is the correct path

st.set_page_config(page_title="AI Career Predictor", page_icon="🚀")

st.title("🎓 AI Career Prediction")
st.markdown("Enter your details to get a career suggestion based on your interests and background.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=20)

hobby = st.selectbox("Select a Hobby", label_encoders['Hobbies'].classes_)
entrepreneurship = st.selectbox("Interested in Entrepreneurship?", label_encoders['Entrepreneurship'].classes_)
favorite_subject = st.selectbox("Favorite Subject", label_encoders['Favorite_Subject'].classes_)

if st.button("Predict Career"):
    try:
        # Encode categorical inputs
        hobby_enc = label_encoders['Hobbies'].transform([hobby])[0]
        entrepreneurship_enc = label_encoders['Entrepreneurship'].transform([entrepreneurship])[0]
        fav_sub_enc = label_encoders['Favorite_Subject'].transform([favorite_subject])[0]

        # Prepare features and predict
        features = np.array([[age, hobby_enc, entrepreneurship_enc, fav_sub_enc]])
        prediction = model.predict(features)[0]

        # Decode prediction
        predicted_career = label_encoders['Predicted_Career_Domain'].inverse_transform([prediction])[0]

        st.success(f"🎯 Suggested Career Path: **{predicted_career}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
