import pandas as pd
import joblib

model = joblib.load('fraud_model.pkl')
print("Model loaded successfully.")

df = pd.read_csv('creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

y_pred = model.predict(X)
print("Prediction done. First 10 predictions:")
print(y_pred[:10])

index = 0
single_transaction = X.iloc[[index]]
result = model.predict(single_transaction)[0]

print(f"Prediction for transaction {index}: ")
print(" Fraud" if result == 1 else " Normal")