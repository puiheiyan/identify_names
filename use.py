import joblib
from helpe import *

# Load the model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Predict with new data
new_names = ["Mary", "John Wick"]
preprocessed_names = pd.concat([preprocess_name(name) for name in new_names])
preprocessed_names_scaled = scaler.transform(preprocessed_names)
predictions = model.predict(preprocessed_names_scaled)
predicted_probabilities = model.predict_proba(preprocessed_names_scaled)[:, 1]

# Output the results
for name, prediction, probability in zip(new_names, predictions, predicted_probabilities):
    print(f"Name: {name}, Prediction: {prediction}, Probability: {probability:.2f}")
    