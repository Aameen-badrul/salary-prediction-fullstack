import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Import our custom function from File 1
from data_preprocessing import load_and_clean_data

def train_and_save_model():
    print("Loading data...")
    X, y, lambda_value = load_and_clean_data('Employers_data.csv')

    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup your Transformers
    education_order = [['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('edu_oe', OrdinalEncoder(categories=education_order), ['Education_Level']),
            ('cat_ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Gender', 'Department', 'Job_Title'])
        ],
        remainder='passthrough'
    )

    # 3. Create the Professional Pipeline
    # This automatically runs the preprocessor, then scales it, then trains the model!
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    print("Training Linear Regression model...")
    model_pipeline.fit(X_train, y_train)

    print(f"Train score (R2): {model_pipeline.score(X_train, y_train):.4f}")
    print(f"Test score (R2): {model_pipeline.score(X_test, y_test):.4f}")

    # 4. Save the Pipeline AND the lambda_value together in a dictionary
    print("Saving model and lambda value to Salary_prediction_model.pkl...")
    model_data = {
        'model': model_pipeline,
        'lambda_value': lambda_value
    }
    
    with open('Salary_prediction_model.pkl', 'wb') as file:
        pickle.dump(model_data, file)
        
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()