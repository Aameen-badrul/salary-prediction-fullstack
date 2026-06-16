from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from scipy.special import inv_boxcox

# Initialize the API
app = FastAPI(title="Employee Salary Prediction API", description="My ML Portfolio Project")

# Load the model exactly once when the server starts
print("Loading model...")
with open('Salary_prediction_model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    
loaded_pipeline = model_data['model']
lambda_value = model_data['lambda_value']

# Define the expected inputs for our API
class EmployeeData(BaseModel):
    Age: int
    Gender: str
    Department: str
    Job_Title: str
    Experience_Years: int
    Education_Level: str

# Create the Endpoint (The URL people will send data to)
@app.post("/predict")
def predict_salary(employee: EmployeeData):
    # Convert the incoming JSON data into a Pandas DataFrame
    input_dict = employee.model_dump() 
    df = pd.DataFrame([input_dict])
    
    # Run the pipeline to get the prediction
    transformed_prediction = loaded_pipeline.predict(df)[0]
    
    # Reverse the Box-Cox math to get real dollars
    real_salary = inv_boxcox(transformed_prediction, lambda_value)
    
    # Return the answer
    return {
        "status": "success",
        "predicted_salary": round(real_salary, 2)
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT
from pydantic import BaseModel
import pickle
import pandas as pd
from scipy.special import inv_boxcox

app = FastAPI(title="Employee Salary Prediction API")

# --- NEW CORS SECURITY RULES ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any frontend to connect (good for local testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)
# -------------------------------

print("Loading model...")
with open('Salary_prediction_model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    
loaded_pipeline = model_data['model']
lambda_value = model_data['lambda_value']

class EmployeeData(BaseModel):
    Age: int
    Gender: str
    Department: str
    Job_Title: str
    Experience_Years: int
    Education_Level: str

@app.post("/predict")
def predict_salary(employee: EmployeeData):
    input_dict = employee.model_dump() 
    df = pd.DataFrame([input_dict])
    transformed_prediction = loaded_pipeline.predict(df)[0]
    real_salary = inv_boxcox(transformed_prediction, lambda_value)
    
    return {
        "status": "success",
        "predicted_salary": round(real_salary, 2)
    }