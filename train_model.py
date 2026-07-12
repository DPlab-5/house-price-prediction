import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import joblib

# Load dataset
df = pd.read_csv('Housing_Data_Train.csv')

# See first 5 rows
print(df.head())

# See basic info
print(df.info())

# # Check missing values
# print(df.isnull().sum())

# Drop rows with missing values
# df = df.dropna()

# Check missing values before cleaning
print("Missing values before cleaning:", df.isnull().sum().sum())

# --- FIX: Fill missing values cleanly without deleting rows ---
# 1. Fill numeric columns with their median values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 2. Fill text columns with 'None'
text_cols = df.select_dtypes(include=['object']).columns
df[text_cols] = df[text_cols].fillna('None')

# 3. Drop ID columns that shouldn't be used for price forecasting
df = df.drop(['Id', 'Unnamed: 0'], axis=1, errors='ignore')

# Convert text column to numbers
df = pd.get_dummies(df)

# Separate features (X) and target price (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

model_columns = X.columns.tolist()

print("Data shape:", X.shape)
print("Cleaned successfully!")

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data (makes all numbers similar range)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Define all 5 models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # This saves the exact list and order of your 297 columns
    # joblib.dump(model_columns, 'model_columns.pkl')
    # print("Successfully saved model columns layout!")
    
    # Make predictions
    # predictions = model.predict(X_test)
    
    # Calculate accuracy scores
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    results[name] = {"RMSE": rmse, "R2 Score": r2, "model": model}
    print(f"{name} → RMSE: {rmse:.2f} | R2 Score: {r2:.4f}")

    # Find best model based on R2 Score (higher is better)
best_model_name = max(results, key=lambda x: results[x]["R2 Score"])
best_model = results[best_model_name]["model"]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"R2 Score: {results[best_model_name]['R2 Score']:.4f}")

# Create comparison chart
model_names = list(results.keys())
r2_scores = [results[m]["R2 Score"] for m in model_names]
rmse_scores = [results[m]["RMSE"] for m in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# R2 Score chart
ax1.bar(model_names, r2_scores, color=['blue','green','orange','red','purple'])
ax1.set_title('Model Comparison - R2 Score (Higher is Better)')
ax1.set_ylabel('R2 Score')
ax1.set_ylim(0, 1)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15)

# RMSE chart
ax2.bar(model_names, rmse_scores, color=['blue','green','orange','red','purple'])
ax2.set_title('Model Comparison - RMSE (Lower is Better)')
ax2.set_ylabel('RMSE')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
print("Chart saved!")

# Save best model as a file
# with open('model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)

# # Save the scaler too (needed later for predictions)
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)


joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

print("✅ Model saved as model.pkl")
print("✅ Scaler saved as scaler.pkl")