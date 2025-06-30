import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")  # Ensure this .pkl file is in the same folder

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to check whether they would have survived.")

# Input fields
with st.form("passenger_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
    fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    submitted = st.form_submit_button("Predict")

# Encode inputs as your training did
def encode_inputs(sex, embarked):
    sex_encoded = 0 if sex.lower() == "male" else 1
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    embarked_encoded = embarked_map.get(embarked.upper(), 0)
    return sex_encoded, embarked_encoded

# Predict when form is submitted
if submitted:
    sex_enc, embarked_enc = encode_inputs(sex, embarked)
    input_data = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
    prediction = model.predict(input_data)[0]
    prediction_text = "🎉 Survived" if prediction == 1 else "💀 Not Survived"

    # Output
    st.subheader("Prediction Result:")
    st.write(prediction_text)
