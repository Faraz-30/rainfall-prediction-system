import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("Rainfall Prediction System")

data = pd.read_csv("weatherAUS.csv")

data = data.dropna()
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes':1,'No':0})

features = [
'MinTemp','MaxTemp','Rainfall','WindGustSpeed',
'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
'Temp9am','Temp3pm','WindSpeed9am','WindSpeed3pm'
]

X = data[features]
y = data['RainTomorrow']

model = RandomForestClassifier()
model.fit(X,y)

st.header("Enter Weather Details")

MinTemp = st.number_input("Min Temperature")
MaxTemp = st.number_input("Max Temperature")
Rainfall = st.number_input("Rainfall")
WindGustSpeed = st.number_input("Wind Gust Speed")
Humidity9am = st.number_input("Humidity 9am")
Humidity3pm = st.number_input("Humidity 3pm")
Pressure9am = st.number_input("Pressure 9am")
Pressure3pm = st.number_input("Pressure 3pm")
Temp9am = st.number_input("Temp 9am")
Temp3pm = st.number_input("Temp 3pm")
WindSpeed9am = st.number_input("Wind Speed 9am")
WindSpeed3pm = st.number_input("Wind Speed 3pm")

if st.button("Predict"):

    input_data = np.array([[MinTemp,MaxTemp,Rainfall,WindGustSpeed,
                            Humidity9am,Humidity3pm,
                            Pressure9am,Pressure3pm,
                            Temp9am,Temp3pm,
                            WindSpeed9am,WindSpeed3pm]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Rain Tomorrow 🌧️")
    else:
        st.success("No Rain Tomorrow ☀️")
