import pandas as pd

# Load datasets
pcos_df = pd.read_csv("PCOS_data.csv")
mental_health_df = pd.read_csv("Indian_mental_wellness.csv")
lifestyle_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Standardize column names (strip spaces, lowercase, and rename relevant ones)
pcos_df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
mental_health_df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
lifestyle_df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

# Rename specific columns for consistency
pcos_df.rename(columns={'age_(yrs)': 'age', 'height(cm)': 'height', 'weight_(kg)': 'weight',
                        'bp__systolic_(mmhg)': 'bp_systolic', 'bp__diastolic_(mmhg)': 'bp_diastolic',
                        'cycle_length(days)': 'cycle_length'}, inplace=True)

lifestyle_df.rename(columns={'sleep_duration': 'sleep_duration', 'stress_level': 'stress_level',
                             'physical_activity_level': 'physical_activity_level'}, inplace=True)

mental_health_df.rename(columns={'awareness_of_mental_health': 'awareness_of_mental_health',
                                 'perception_of_therapy': 'perception_of_therapy'}, inplace=True)

# Add gender column to PCOS dataset (since it only contains female data)
pcos_df["gender"] = "Female"

# Filter only female records in mental health and lifestyle datasets
mental_health_df = mental_health_df[mental_health_df["gender"].str.lower() == "female"]
lifestyle_df = lifestyle_df[lifestyle_df["gender"].str.lower() == "female"]

# Select necessary columns from each dataset
pcos_selected = pcos_df[['age', 'height', 'weight', 'bmi', 'bp_systolic', 'bp_diastolic', 'cycle_length', 'gender']]
mental_health_selected = mental_health_df[['age', 'gender', 'awareness_of_mental_health', 'perception_of_therapy']]
lifestyle_selected = lifestyle_df[['age', 'gender', 'sleep_duration', 'stress_level', 'physical_activity_level']]

# Merge datasets on 'age' and 'gender'
merged_df = pcos_selected.merge(mental_health_selected, on=['age', 'gender'], how='outer')\
                         .merge(lifestyle_selected, on=['age', 'gender'], how='outer')

# Handle missing values by filling with median values
merged_df.fillna(merged_df.median(numeric_only=True), inplace=True)

# Save merged dataset
merged_df.to_csv("merged_dataset.csv", index=False)

print("Merged dataset saved as merged_dataset.csv")
