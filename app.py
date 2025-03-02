from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Load trained model & scaler
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")

# Define risk categories
risk_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.json
        user_input = [
            data["age"], data["height"], data["weight"], data["bmi"],
            data["bp_systolic"], data["bp_diastolic"], data["cycle_length"],
            0,  # Gender default as Female
            data["awareness_of_mental_health"], data["perception_of_therapy"],
            data["sleep_duration"], data["stress_level"], data["physical_activity_level"]
        ]

        # Convert to DataFrame & Scale
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)

        # Predict Cluster
        cluster = kmeans.predict(user_scaled)[0]
        risk_category = risk_labels[cluster]

        return jsonify({"pcos_risk": risk_category})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)