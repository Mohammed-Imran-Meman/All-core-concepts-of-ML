import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X,y = make_classification(n_samples=10000, n_features=10,n_informative=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

print("Decision Tree accuracy",accuracy_score(y_test,y_pred))

# Bagging
bag = BaggingClassifier(
  estimator=DecisionTreeClassifier(),
  n_estimators=500,
  max_samples=0.5,
  bootstrap=True,
  random_state=42
)
     
bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
     
accuracy_score(y_test,y_pred)

bag.estimators_samples_[0].shape
bag.estimators_features_[0].shape

# Using SVM
bag = BaggingClassifier(
  estimator=SVC(),
  n_estimators=500,
  max_samples=0.25,
  bootstrap=True,
  random_state=42
)
     
bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Bagging using SVM",accuracy_score(y_test,y_pred))

# Pasting
bag = BaggingClassifier(
  estimator=DecisionTreeClassifier(),
  n_estimators=500,
  max_samples=0.25,
  bootstrap=False,
  random_state=42,
  verbose = 1,
  n_jobs=-1
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Pasting classifier",accuracy_score(y_test,y_pred))

# Random Subspaces
bag = BaggingClassifier(
  estimator=DecisionTreeClassifier(),
  n_estimators=500,
  max_samples=1.0,
  bootstrap=False,
  max_features=0.5,
  bootstrap_features=True,
  random_state=42
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Random Subspaces classifier",accuracy_score(y_test,y_pred))

bag.estimators_samples_[0].shape
bag.estimators_features_[0].shape

# Random patches
bag = BaggingClassifier(
  estimator=DecisionTreeClassifier(),
  n_estimators=500,
  max_samples=0.25,
  bootstrap=True,
  max_features=0.5,
  bootstrap_features=True,
  random_state=42
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print("Random Patches classifier",accuracy_score(y_test,y_pred))

# OOB Score
bag = BaggingClassifier(
  estimator=DecisionTreeClassifier(),
  n_estimators=500,
  max_samples=0.25,
  bootstrap=True,
  oob_score=True,
  random_state=42
)   

bag.fit(X_train,y_train)
bag.oob_score_

y_pred = bag.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))

# Applying GridSearch CV
parameters = {
  'n_estimators': [50,100,500], 
  'max_samples': [0.1,0.4,0.7,1.0],
  'bootstrap' : [True,False],
  'max_features' : [0.1,0.4,0.7,1.0]
  }
     
search = GridSearchCV(BaggingClassifier(), parameters, cv=5, n_jobs=-1, verbose=2)
search.fit(X_train,y_train)
search.best_params_
search.best_score_