from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Encode categorical variables
label_encoders = {}
for col in ['gender', 'perception_of_therapy']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save the encoded dataset
df.to_csv("encoded_dataset.csv", index=False)
print("Categorical variables encoded and dataset saved as encoded_dataset.csv")

from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Select numerical columns for normalization
num_cols = ['age', 'height', 'weight', 'bmi', 'bp_systolic', 'bp_diastolic', 
            'cycle_length', 'sleep_duration', 'stress_level', 'physical_activity_level']

# Apply normalization
df[num_cols] = scaler.fit_transform(df[num_cols])

# Verify normalization
print(df.head())

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns
df['gender'] = label_encoder.fit_transform(df['gender'])  # Female -> 0
df['awareness_of_mental_health'] = label_encoder.fit_transform(df['awareness_of_mental_health'])  
df['perception_of_therapy'] = label_encoder.fit_transform(df['perception_of_therapy'])  

# Verify encoding
print(df.head())

from sklearn.preprocessing import StandardScaler
# Initialize StandardScaler
scaler = StandardScaler()

# List of numerical columns to scale
numerical_cols = ['age', 'height', 'weight', 'bmi', 'bp_systolic', 'bp_diastolic',
                  'cycle_length', 'sleep_duration', 'stress_level', 'physical_activity_level']

# Apply standardization
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Verify scaling
print(df.head())

# Initialize StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical columns
scaled_features = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=df.columns)

# Display first few rows
print(df_scaled.head())
