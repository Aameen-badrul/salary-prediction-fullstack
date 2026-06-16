import pickle
import pandas as pd
from scipy.special import inv_boxcox

def make_prediction(input_data):
    # 1. Load the saved model data
    with open('Salary_prediction_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
        
    loaded_pipeline = model_data['model']
    lambda_value = model_data['lambda_value']
    
    # 2. Convert the user input into a DataFrame
    df = pd.DataFrame([input_data])
    
    # 3. Make the prediction using the pipeline 
    # (The pipeline automatically handles the OneHotEncoding and Scaling for us!)
    transformed_prediction = loaded_pipeline.predict(df)[0]
    
    # 4. Reverse the Box-Cox transformation to get a real Dollar amount
    real_salary = inv_boxcox(transformed_prediction, lambda_value)
    
    return real_salary

if __name__ == "__main__":
    # Let's test it with a fake new employee
    sample_employee = {
        'Age': 30,
        'Gender': 'Male',
        'Department': 'Engineering',
        'Job_Title': 'Manager',
        'Experience_Years': 8,
        'Education_Level': 'Master'
    }
    
    predicted_salary = make_prediction(sample_employee)
    print(f"Predicted Salary for this employee: ${predicted_salary:,.2f}")