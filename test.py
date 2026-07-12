import joblib

model_columns = joblib.load("model_columns.pkl")

print(model_columns[:5])