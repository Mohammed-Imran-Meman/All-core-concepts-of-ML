import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from ridge_code_of_2d_regression import MyRidge

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_

r = Ridge(alpha=10)
r.fit(X_train,y_train)
r.coef_
r.intercept_

r2 = Ridge(alpha=100)
r2.fit(X_train,y_train)
r2.coef_
r2.intercept_

plt.plot(X,y,'b.')
plt.plot(X,lr.predict(X),color='red',label='alpha=0')
plt.plot(X,r.predict(X),color='green',label='alpha=10')
plt.plot(X,r2.predict(X),color='orange',label='alpha=100')
plt.legend()
# getting Ridge model from the file of the code of 2D linear regression

my_ridge = MyRidge(alpha=10)
my_ridge.fit(X_train,y_train)
my_ridge.m
my_ridge.b

my_ridge = MyRidge(alpha=100)
my_ridge.fit(X_train,y_train)
my_ridge.m
my_ridge.b