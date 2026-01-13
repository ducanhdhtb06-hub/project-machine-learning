Built a classification model to predict CSGO match outcomes based on match statistics and contextual features.

Performed feature selection by removing time-based and non-informative attributes to improve model efficiency and focus.

Designed a complete data preprocessing pipeline using Pipeline and ColumnTransformer to handle mixed feature types.

Applied missing value imputation using SimpleImputer (median strategy for numerical features, constant value for categorical features).

Scaled numerical features with StandardScaler to ensure balanced feature contribution.
Encoded categorical feature (map) using OneHotEncoder.

Trained a Random Forest Classifier and optimized hyperparameters using GridSearchCV with 6-fold cross-validation.

Used precision_weighted as the evaluation metric to address potential class imbalance.

Evaluated model performance using classification report (precision, recall, F1-score).

Gained hands-on experience in building an end-to-end Machine Learning classification pipeline, from preprocessing to model tuning and evaluation.
