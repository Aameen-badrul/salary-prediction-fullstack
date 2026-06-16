# 🚀 Employee Salary Prediction (Full-Stack ML App)
**Author:** Mohammad Badrul

## 📌 Overview
This is a full-stack Machine Learning application that predicts an employee's salary based on features like Age, Gender, Department, Job Title, Years of Experience, and Education Level. 

Unlike standard data science scripts, this project is built for production: it uses a scikit-learn `Pipeline` to handle data preprocessing and is served via a REST API using **FastAPI**. It includes a lightweight HTML/JS frontend for real-time user interaction.

## 🛠️ Tech Stack
* **Machine Learning:** Python, scikit-learn, pandas, scipy (Box-Cox Transformation)
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## 🧠 Machine Learning Details
* **Data Preprocessing:** Handled categorical variables using `OneHotEncoder` and `OrdinalEncoder`. Normalized numerical features using `StandardScaler`.
* **Target Transformation:** Applied Box-Cox transformation to the skewed Salary data to improve model accuracy, saving the lambda value for inverse transformations in the API.
* **Model:** Multiple Linear Regression achieving an R2 score of ~0.99.

## 🚀 How to Run Locally

**1. Clone the repository:**
\`\`\`bash
 git clone https://github.com/Aameen-badrul/salary-prediction-fullstack.git
cd your-repo-name
\`\`\`

**2. Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**3. Start the FastAPI Backend Server:**
\`\`\`bash
python -m uvicorn app:app --reload
\`\`\`

**4. Open the Frontend:**
Simply double-click the `index.html` file to open it in your web browser and test the model! You can also test the API directly by visiting `http://127.0.0.1:8000/docs`.