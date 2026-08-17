import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)


# -------------------------
# 1. Dataset
# -------------------------

df = pd.read_csv("data.csv")


# -------------------------
# 2. Inspect Dataset
# -------------------------

print(df.head())

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistics:")
print(df.describe())


# -------------------------
# 3. Separate Features
# -------------------------

X = df.drop(
    columns=["target"]
)

y = df["target"]


# -------------------------
# 4. Identify Feature Types
# -------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns

categorical_features = X.select_dtypes(
    exclude=np.number
).columns


# -------------------------
# 5. Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# 6. Handle Missing Values
# -------------------------

numeric_imputer = SimpleImputer(
    strategy="mean"
)

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)


X_train[numeric_features] = numeric_imputer.fit_transform(
    X_train[numeric_features]
)

X_test[numeric_features] = numeric_imputer.transform(
    X_test[numeric_features]
)


X_train[categorical_features] = categorical_imputer.fit_transform(
    X_train[categorical_features]
)

X_test[categorical_features] = categorical_imputer.transform(
    X_test[categorical_features]
)


# -------------------------
# 7. Encode Categorical Features
# -------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

if len(categorical_features) > 0:

    X_train_encoded = encoder.fit_transform(
        X_train[categorical_features]
    )

    X_test_encoded = encoder.transform(
        X_test[categorical_features]
    )

else:

    X_train_encoded = np.empty(
        (len(X_train), 0)
    )

    X_test_encoded = np.empty(
        (len(X_test), 0)
    )


# -------------------------
# 8. Convert Numerical Features
# -------------------------

X_train_numeric = X_train[
    numeric_features
].to_numpy()

X_test_numeric = X_test[
    numeric_features
].to_numpy()


# -------------------------
# 9. Feature Scaling
# -------------------------

scaler = StandardScaler()

X_train_numeric = scaler.fit_transform(
    X_train_numeric
)

X_test_numeric = scaler.transform(
    X_test_numeric
)


# -------------------------
# 10. Combine Features
# -------------------------

X_train = np.hstack(
    [
        X_train_numeric,
        X_train_encoded
    ]
)

X_test = np.hstack(
    [
        X_test_numeric,
        X_test_encoded
    ]
)


# -------------------------
# 11. Target Preprocessing
# -------------------------

# Binary / multiclass classification:
# y = LabelEncoder().fit_transform(y)

# Regression:
# y = y.to_numpy()


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# -------------------------
# 12. Final Dataset
# -------------------------

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nTraining Features:")
print(X_train)

print("\nTesting Features:")
print(X_test)

print("\nTraining Target:")
print(y_train)

print("\nTesting Target:")
print(y_test)
