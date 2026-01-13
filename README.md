# Diabetes Prediction using Machine Learning

## Project Overview
This project builds a binary classification model to predict whether a patient has diabetes based on medical features using the Pima Indians Diabetes Dataset.

## Objective
- Apply machine learning techniques to solve a healthcare classification problem.
- Understand the full machine learning workflow from data preprocessing to model evaluation.

## Dataset
- **Dataset:** Pima Indians Diabetes Dataset
- **Target variable:** Outcome
  - 0: Non-diabetic
  - 1: Diabetic
- **Features:** Glucose, Blood Pressure, BMI, Age, Insulin, Pregnancies, etc.

## Methodology
- Split the dataset into training and testing sets.
- Applied feature scaling using **StandardScaler**.
- Trained a **Random Forest Classifier**.
- Optimized hyperparameters using **GridSearchCV (6-fold cross-validation)**.
- Evaluated the model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
