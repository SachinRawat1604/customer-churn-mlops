import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # For saving the model
import os


# ----- Load Data -----
print("Loading data....")
df = pd.read_csv('C:/Users/sachi/OneDrive/Desktop/customer-churn-mlops/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ----- Preprocessing -----
print("Preprocessing data....")

# Convert 'TotalCharges' to numeric, coercing to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# convert 'Churn' from Yes/No to 1/0
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# drop customerID and prepare feature/target
df = df.drop('customerID', axis=1)
X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']

# ----- Train-Test Split -----
print("Splitting data.....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Model Training -----
print("Training model.....")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----- Evaluation -----
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# ----- Save model -----
print("Saving model.....")
#create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_model.pkl')
print("Model saved to models/churn_model.pkl")