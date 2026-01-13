📊 Student Math Score Prediction using Machine Learning
📌 Project Overview

This project builds a regression model to predict students’ math scores based on academic performance and demographic features.

🎯 Objective

Apply machine learning techniques to solve an educational regression problem.

Practice building a complete end-to-end regression pipeline, including preprocessing, model selection, tuning, and evaluation.

📊 Dataset

Dataset: Student Performance Dataset

Target variable: math score

Features:
Reading score, Writing score, Gender, Parental level of education, Lunch, Test preparation course, Race/Ethnicity.

⚙️ Methodology

Split the dataset into training and testing sets.

Built a preprocessing pipeline using Pipeline and ColumnTransformer to handle mixed data types:

Applied SimpleImputer (median strategy) for numerical features.

Applied SimpleImputer (most frequent strategy) for categorical features.

Scaled numerical features (reading score, writing score) using StandardScaler to improve model convergence.

Encoded categorical variables using:

OrdinalEncoder for ordered features (parental level of education, gender, lunch, test preparation course).

OneHotEncoder for nominal features (race/ethnicity).

Used LazyRegressor to quickly compare multiple regression models and identify suitable algorithms.

Implemented SGDRegressor within a preprocessing pipeline.

Optimized hyperparameters using GridSearchCV with R² score as the evaluation metric.

Evaluated model performance using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² score

🧰 Technologies Used

Python

Pandas

NumPy

Scikit-learn

LazyPredict

▶️ How to Run
pip install pandas numpy scikit-learn lazypredict
python main.py

📈 Results

The optimized regression model achieved reliable performance in predicting students’ math scores.

Feature preprocessing and proper encoding significantly improved model stability.

R² score indicates the model’s ability to explain variance in student performance.

📚 Key Learnings

Learned how to preprocess datasets with mixed feature types.

Gained experience in regression model selection and evaluation.

Strengthened understanding of applying machine learning techniques to education-related data.
