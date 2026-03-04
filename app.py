import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("rainfall_model.pkl","rb"))

st.title("Rainfall Prediction System")

st.write("Enter weather details")

MinTemp = st.number_input("Min Temperature")
MaxTemp = st.number_input("Max Temperature")
Rainfall = st.number_input("Rainfall")
WindGustSpeed = st.number_input("Wind Gust Speed")
Humidity9am = st.number_input("Humidity 9am")
Humidity3pm = st.number_input("Humidity 3pm")
Pressure9am = st.number_input("Pressure 9am")
Pressure3pm = st.number_input("Pressure 3pm")
Temp9am = st.number_input("Temperature 9am")
Temp3pm = st.number_input("Temperature 3pm")
WindSpeed9am = st.number_input("Wind Speed 9am")
WindSpeed3pm = st.number_input("Wind Speed 3pm")

if st.button("Predict Rainfall"):

    data = np.array([[MinTemp,MaxTemp,Rainfall,WindGustSpeed,
                      Humidity9am,Humidity3pm,
                      Pressure9am,Pressure3pm,
                      Temp9am,Temp3pm,
                      WindSpeed9am,WindSpeed3pm]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Rain Tomorrow 🌧️")
    else:
        st.success("No Rain Tomorrow ☀️")