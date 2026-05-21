import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import r2_score
from ridge_code import MyRidge

X,y = load_diabetes(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

reg = SGDRegressor(penalty='l2',max_iter=500,eta0=0.1,learning_rate='constant',alpha=0.001)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("R2 score",r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)

r = Ridge(alpha=0.001, max_iter=500,solver='sparse_cg')
r.fit(X_train,y_train)
y_pred1 = r.predict(X_test)
print("R2 score",r2_score(y_test,y_pred1))
r.coef_
r.intercept_

# getting Ridge model from the file of the code of ridge regulation for GD

my_ridge = MyRidge(epochs=500,alpha=0.001,learning_rate=0.005)
my_ridge.fit(X_train,y_train)
y_pred2 = my_ridge.predict(X_test)
print("R2 score",r2_score(y_test,y_pred2))
my_ridge.coef_
my_ridge.intercept_