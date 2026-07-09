import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv('Training.csv')   # change name if needed
test_data = pd.read_csv('Testing.csv')

# Split features and target
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']

X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'trained_model.joblib')

print("Model trained and saved successfully!")