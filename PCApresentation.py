import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data Loading
file_path = 'features_extracted.csv'  # Update with your file path
data = pd.read_csv(file_path)

sampled_data = data.sample(n=200, random_state=42)  

# Feature Extraction
feature_columns = data.columns[:-2]
subset_data = sampled_data[feature_columns]

# Standardize the Data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(subset_data)

cov_matrix = np.cov(standardized_data.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sorting Eigenvalues and Eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]  
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

n_components = 25
top_eigenvectors = sorted_eigenvectors[:, :n_components]

# Eigenvalues and Explained Variance
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)  # Eigenvalue/sum of eigenvalues
cumulative_variance = np.cumsum(explained_variance)

# Plot the Scree Plot with larger fonts
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_components + 1), explained_variance[:n_components], alpha=0.7, label='Explained Variance')
plt.plot(range(1, n_components + 1), cumulative_variance[:n_components], marker='o', color='red', label='Cumulative Variance')

# Set larger font sizes
plt.xlabel("Principal Component Index", fontsize=14)
plt.ylabel("Variance Explained", fontsize=14)
plt.title("Scree Plot and Cumulative Explained Variance", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)  # Legend font size
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.show()

# Print the most contributing feature for the first 15 principal components with larger text
for i in range(15):
    contributions = np.abs(top_eigenvectors[:, i])  # Contributions for the i-th PC
    most_contributing_index = np.argmax(contributions)  # Index of the max contribution
    most_contributing_column = feature_columns[most_contributing_index]
    print(f"\033[1mFeature with max contribution to Principal Component {i + 1}: {most_contributing_column}\033[0m")
