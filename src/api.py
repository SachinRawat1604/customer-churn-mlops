import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI(title = 'Customer Churn Prediction API', version='1.0')

# Load the trained model
# Note: The path is relative to the project's root directory
model = joblib.load('models/churn_model.pkl')

# Define the request body structure using Pydantic
# This ensures that we receive the data in the correct format

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: int
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contact: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Define the root endpoint
@app.get("/")
def read_root():
    return{"message": "Welcome to the Customer Churn Prediction API"}

# define the prediction endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    # convert the incoming data into a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # One-hot encode the categorical features
    input_df = pd.get_dummies(input_df)

    # The model was trained on a specific set of columns
    # We need make sure the input data has the same columns
    # Get the feature names the model was trained on (excluding the target)
    train_features = model.feature_names_in_

    # Align the columns of the input data with the training data
    # Missing columns will be added and filled with 0 (since they dummy variables)
    input_df_aligned = input_df.reindex(columns=train_features, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_df_aligned)
    probability = model.predict_proba(input_df_aligned)[:, 1] # Probability of churn

    # Return the result
    return {
        "prediction": "Churn" if prediction[0] == 1 else "No Churn",
        "churn_probability": float(probability[0])
    }