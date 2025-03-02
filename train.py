import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the merged dataset
file_path = "encoded_dataset.csv"  # Update if needed
df = pd.read_csv(file_path)

# Select features for clustering
features = [
    "age", "height", "weight", "bmi", "bp_systolic", "bp_diastolic",
    "cycle_length", "gender", "awareness_of_mental_health", "perception_of_therapy",
    "sleep_duration", "stress_level", "physical_activity_level"
]

# Extract relevant data
df_selected = df[features]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Train KMeans model with 3 clusters (Low, Moderate, High Risk)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df_scaled)

# Save the scaler and trained model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans.pkl")

print("✅ Model trained & saved successfully!")
