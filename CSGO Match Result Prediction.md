🎮 CSGO Match Result Prediction using Machine Learning
📌 Project Overview

This project builds a classification model to predict the outcome of CSGO matches based on match statistics and contextual features.

🎯 Objective

Apply machine learning techniques to predict competitive game outcomes.

Practice building a complete end-to-end classification pipeline with preprocessing and model tuning.

Handle mixed data types and potential class imbalance.

📊 Dataset

Dataset: CSGO match statistics dataset

Target variable: result

Win / Loss

Features:
Match statistics, map information, team-related numerical features, and contextual attributes.

⚙️ Methodology

Removed time-based and non-informative features (date, time, ping, rounds, etc.) to improve model focus.

Split the dataset into training and testing sets.

Built a preprocessing pipeline using Pipeline and ColumnTransformer:

Applied SimpleImputer (median strategy) for numerical features.

Applied SimpleImputer (constant value) for categorical features.

Scaled numerical features using StandardScaler.

Encoded categorical feature (map) using OneHotEncoder.

Trained a Random Forest Classifier.

Optimized hyperparameters using GridSearchCV with 6-fold cross-validation.

Used precision_weighted as the evaluation metric to handle potential class imbalance.

Evaluated model performance using:

Precision

Recall

F1-score

Classification Report

🧰 Technologies Used

Python

Pandas

NumPy

Scikit-learn

▶️ How to Run
pip install pandas numpy scikit-learn
python main.py

📈 Results

The optimized Random Forest model achieved stable predictive performance.

Weighted precision helped ensure balanced evaluation across match outcomes.

Feature preprocessing significantly improved model robustness.

📚 Key Learnings

Learned how to design a machine learning pipeline for mixed data types.

Gained experience in feature selection, preprocessing, and model optimization.

Improved understanding of applying machine learning to game analytics and esports data.
