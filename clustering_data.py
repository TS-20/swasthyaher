import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv("encoded_dataset.csv")  # Replace with your actual filename

from sklearn.cluster import KMeans

# Select the number of clusters (change as needed)
num_clusters = 3  

# Fit K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df)  # Assign clusters to the dataset

df.to_csv("clustered_dataset.csv", index=False)  # Save the dataset with clusters