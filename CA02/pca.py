import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("datasets/deep_learning_lab_test_dataset.csv")

print(data.head())
print("Shape:", data.shape)

# Encode categorical columns
data = pd.get_dummies(data, dtype=float)

# Separate features and target
features = data.drop("target", axis=1).values
target = data["target"].values

# Handle missing values
features = np.nan_to_num(features)

# Standardize
features = StandardScaler().fit_transform(features)

# Split data
train_features, test_features, train_target, test_target = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

print("Train:", train_features.shape)
print("Test:", test_features.shape)


# Step 1: Mean centering
mean = np.mean(train_features, axis=0)

centered_train = train_features - mean
centered_test = test_features - mean


# Step 2: Covariance matrix
covariance = np.cov(centered_train, rowvar=False)

print("\nCovariance Matrix:")
print(covariance)


# Step 3: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance)


# Step 4: Sort in descending order
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]


# Step 5: Select principal components
components = eigenvectors[:, :2]


# Step 6: Transform data
pca_train = centered_train @ components
pca_test = centered_test @ components

print("\nOriginal shape:", train_features.shape)
print("PCA shape:", pca_train.shape)


# Explained variance
explained_variance = eigenvalues / eigenvalues.sum()

print("\nEigenvalues:")
print(eigenvalues)

print("\nExplained Variance:")
print(explained_variance)

print("\nTotal Explained Variance:",
      explained_variance[:2].sum())


# Plot
plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

plt.plot(np.cumsum(explained_variance))
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("PCA Explained Variance")
plt.show()
