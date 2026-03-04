import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("weatherAUS.csv")

# Remove missing values
data = data.dropna()

# Convert Yes/No to 1/0
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes':1,'No':0})

# Features for prediction
features = [
'MinTemp','MaxTemp','Rainfall','WindGustSpeed',
'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
'Temp9am','Temp3pm','WindSpeed9am','WindSpeed3pm'
]

X = data[features]
y = data['RainTomorrow']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("rainfall_model.pkl", "wb"))

print("Model saved as rainfall_model.pkl")