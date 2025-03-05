# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("traffic volume.csv")  # Update with actual file path
print("✅ Dataset Loaded Successfully!\n")

# Display first 5 rows
print("📊 First 5 Rows of Data:\n", df.head())

# Summary statistics
print("\n📊 Dataset Summary:\n", df.describe())

# Dataset information
print("\n📊 Dataset Info:")
df.info()

# Convert "date" column to datetime format
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# Extract date-related features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Convert "Time" column to hour
df["hour"] = pd.to_datetime(df["Time"]).dt.hour

# Drop unnecessary columns
df.drop(["date", "Time"], axis=1, inplace=True)

# ✅ Handling Missing Values & Incorrect Zeros
# Replace '0' in 'rain' and 'snow' with NaN (assuming they represent missing data)
df["rain"].replace(0, np.nan, inplace=True)
df["snow"].replace(0, np.nan, inplace=True)

# Convert 'None' (string) in 'holiday' column to NaN before handling missing values
df["holiday"].replace("None", np.nan, inplace=True)

# Check for missing values before handling
print("\n🔍 Checking Missing Values Before Handling:\n", df.isnull().sum())

# Fill missing numeric values (temp, rain, snow) with the column mean
numeric_cols = ["temp", "rain", "snow"]
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing categorical values (weather) with the most frequent value (mode)
df["weather"].fillna(df["weather"].mode()[0], inplace=True)  

# Fill missing values in holiday column back to 'None'
df["holiday"].fillna("None", inplace=True)

# Verify missing values after handling
print("\n🔍 Checking Missing Values After Handling:\n", df.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_features = ["holiday", "weather"]

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# Define Features and Target Variable
feature_columns = ["temp", "rain", "snow", "year", "month", "day", "hour", "holiday", "weather"]
target_column = "traffic_volume"

X = df[feature_columns]  # Features
y = df[target_column]    # Target Variable

# Split into Train/Test Sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot of Features")
plt.show()

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Train models & Evaluate Performance
evaluation_results = []
best_model = None
best_score = float("-inf")

for name, model in models.items():
    model.fit(X_train, y_train)  # Train Model
    print(f"✅ {name} model trained successfully!")

    y_pred = model.predict(X_test)  # Make Predictions

    # Compute Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    evaluation_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    # Update Best Model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# Convert results into DataFrame
df_results = pd.DataFrame(evaluation_results)

# Display Model Performance Comparison
print("\n📊 Model Performance Comparison:\n", df_results)

# Print Best Model
print(f"\n🎯 Best Model: {best_model_name} with R² Score: {best_score:.4f}")

# Save the best model
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save label encoders
for col, encoder in label_encoders.items():
    pickle.dump(encoder, open(f"{col}_encoder.pkl", "wb"))

print("✅ Model and encoders saved successfully!")
