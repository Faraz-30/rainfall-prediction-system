import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Rainfall Prediction System 🌧️")

# Load dataset
data = pd.read_csv("weatherAUS.csv")

# Clean dataset
data = data.dropna(subset=['RainTomorrow'])
data = data.fillna(data.median(numeric_only=True))

# Convert Yes/No to numbers
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes':1,'No':0})
data['RainToday'] = data['RainToday'].map({'Yes':1,'No':0})

# Important features
features = [
'MinTemp','MaxTemp','Rainfall','WindGustSpeed',
'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
'Temp9am','Temp3pm','WindSpeed9am','WindSpeed3pm','RainToday'
]

X = data[features]
y = data['RainTomorrow']

# Train model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train,y_train)

# Accuracy display
pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)

st.write("Model Accuracy:", round(accuracy*100,2), "%")

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
RainToday = st.selectbox("Rain Today", [0,1])

if st.button("Predict Rainfall"):

    input_data = np.array([[MinTemp,MaxTemp,Rainfall,WindGustSpeed,
                            Humidity9am,Humidity3pm,
                            Pressure9am,Pressure3pm,
                            Temp9am,Temp3pm,
                            WindSpeed9am,WindSpeed3pm,
                            RainToday]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Rain Tomorrow 🌧️")
    else:
        st.success("No Rain Tomorrow ☀️")
