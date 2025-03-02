import joblib
import numpy as np

# Load the trained scaler and KMeans model
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")

# Define function to predict PCOS risk
def predict_pcos_risk(user_input, scaler, kmeans):
    user_array = np.array(user_input).reshape(1, -1)  # Convert input to array
    user_scaled = scaler.transform(user_array)  # Scale the input
    cluster = kmeans.predict(user_scaled)[0]  # Predict the cluster

    # Map clusters to risk levels
    risk_mapping = {0: "Moderate Risk", 1: "Low Risk", 2: "Higher Risk"}
    
    return risk_mapping[cluster]

# Take user input
print("Enter your details:")
age = float(input("Age: "))
height = float(input("Height (cm): "))
weight = float(input("Weight (kg): "))
bmi = weight / ((height / 100) ** 2)  # Calculate BMI
bp_systolic = float(input("BP Systolic (mmHg): "))
bp_diastolic = float(input("BP Diastolic (mmHg): "))
cycle_length = float(input("Cycle Length (days): "))
awareness_of_mental_health = float(input("Awareness of Mental Health (0 or 1): "))
perception_of_therapy = float(input("Perception of Therapy (Positive=1, Negative=0): "))
sleep_duration = float(input("Sleep Duration (hours): "))
stress_level = float(input("Stress Level (1-10): "))
physical_activity_level = float(input("Physical Activity Level (minutes per day): "))

# Set gender to Female (0) by default
gender = 0  

# Store input in list
user_data = [
    age, height, weight, bmi, bp_systolic, bp_diastolic, cycle_length,
    gender, awareness_of_mental_health, perception_of_therapy,
    sleep_duration, stress_level, physical_activity_level
]

# Predict risk category
predicted_risk = predict_pcos_risk(user_data, scaler, kmeans)

# Output the result
print(f"\n✅ Based on your input, your **PCOS Risk Level** is: **{predicted_risk}**")
