import pandas as pd
from scipy.stats import boxcox

def load_and_clean_data(file_path):
    # 1. Load the data
    df = pd.read_csv(file_path)
    
    # 2. Drop the columns you specified
    df = df.drop(['Name', 'Location', 'Employee_ID'], axis=1)
    
    # 3. Handle the target variable using your Box-Cox logic
    df['Salary_boxcox'], lambda_value = boxcox(df['Salary'])
    
    # 4. Split into Features (X) and Target (y)
    # Dropping both raw salary and transformed salary from features
    X = df.drop(['Salary', 'Salary_boxcox'], axis=1)
    y = df['Salary_boxcox']
    
    return X, y, lambda_value