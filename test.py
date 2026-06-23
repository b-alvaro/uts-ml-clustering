import joblib

model = joblib.load("kmeans_model.pkl")

print(type(model))