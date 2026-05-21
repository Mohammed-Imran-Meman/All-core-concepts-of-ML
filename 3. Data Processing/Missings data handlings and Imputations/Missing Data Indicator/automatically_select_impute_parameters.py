import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv')
df.head()
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
df.head()
X = df.iloc[:,1:]
y = df['Survived']
df.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

trf = ColumnTransformer([
    ('num', num_pipe, ['Age', 'Fare']),
    ('cat', cat_pipe, ['Sex', 'Embarked'])
], remainder='passthrough')

pipe = Pipeline([
    ('trf', trf),
    ('lr', LogisticRegression())
])

param_grid = {
    'trf__num__impute__strategy': ['mean', 'median'],
    'trf__cat__impute__strategy': ['most_frequent', 'constant'],
    'lr__C': [0.1, 1.0, 10, 100]
}

grid_search = GridSearchCV(pipe, param_grid, cv=10)

grid_search.fit(X_train, y_train)

print(f"Best params:")
print(grid_search.best_params_)

print(f"Internal CV score: {grid_search.best_score_:.3f}")

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[['param_lr__C','param_trf__cat__impute__strategy','param_trf__num__impute__strategy','mean_test_score']]