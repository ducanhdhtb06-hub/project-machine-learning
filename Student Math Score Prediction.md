Built a regression model to predict students’ math scores based on academic performance and demographic features.

Performed data preprocessing using Pipeline and ColumnTransformer to handle mixed data types (numerical, ordinal, and nominal).

Applied missing value imputation using SimpleImputer:

Median strategy for numerical features.

Most frequent strategy for categorical features.

Scaled numerical features (reading score, writing score) using StandardScaler to improve model convergence.

Encoded categorical variables using:

OrdinalEncoder for ordered features (parental level of education, gender, lunch, test preparation course).

OneHotEncoder for nominal features (race/ethnicity).
Used LazyRegressor to quickly compare multiple regression models and identify suitable algorithms.

Implemented SGDRegressor within a preprocessing pipeline.

Optimized hyperparameters using GridSearchCV with R² score as the evaluation metric.

Evaluated model performance using MAE, MSE, and R² score to assess prediction accuracy and goodness of fit.

Gained hands-on experience in building an end-to-end Machine Learning regression pipeline, from data preprocessing to model selection, tuning, and evaluation.
