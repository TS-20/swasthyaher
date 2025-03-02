import pandas as pd

# Load dataset
df = pd.read_csv("merged_dataset.csv")

# Handling missing values
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("Missing values handled and dataset saved as cleaned_dataset.csv")

