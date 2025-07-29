import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('creditcard.csv')
print(data.head(3))

print(data['Class'].value_counts())

print("Number of fraud transactioon:",len(data[data['Class'] == 1]))
print("Number of valid transaction:",len(data[data['Class'] == 0]))

X = data.drop('Class',axis=1)
y = data['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

joblib.dump(model, 'model.pkl')
print(" Model saved as model.pkl")
