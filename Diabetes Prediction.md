🩺 Diabetes Prediction using Machine Learning
📌 Project Overview

This project builds a binary classification model to predict whether a patient has diabetes based on medical features using the Pima Indians Diabetes Dataset.

🎯 Objective

Apply machine learning techniques to solve a healthcare classification problem.

Practice the end-to-end machine learning workflow, from data preprocessing to model tuning and evaluation.

📊 Dataset

Dataset: Pima Indians Diabetes Dataset

Target variable: Outcome

0: Non-diabetic

1: Diabetic

Features:
Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.

⚙️ Methodology

Split the dataset into training and testing sets.

Applied feature scaling using StandardScaler to improve model convergence.

Trained a Random Forest Classifier.

Optimized hyperparameters using GridSearchCV with 6-fold cross-validation.

Evaluated model performance using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🧰 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

▶️ How to Run
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py

📈 Results

The optimized Random Forest model achieved stable classification performance.

Hyperparameter tuning improved generalization on unseen data.

Evaluation metrics indicate good predictive capability for diabetes detection.

📚 Key Learnings

Learned how to preprocess medical data for machine learning models.

Gained experience in feature scaling, model tuning, and classification evaluation.

Strengthened understanding of applying machine learning techniques to healthcare datasets.
