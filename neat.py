import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from helpe import *
import joblib


# Sample data
df = pd.read_csv("lender.csv")

# Feature selection
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=features, inplace=True)

X = df[features]
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Average cross-validation score: {cv_scores.mean()}")

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = model.score(X_test_scaled, y_test)
# print(f"Model accuracy: {accuracy:.2f}")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)

joblib.dump(model, 'logistic_regression_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
