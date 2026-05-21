import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

X = 6 * np.random.rand(200,1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200,1)

plt.scatter(X,y)
plt.xlabel('X')
plt.ylabel('y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)

plt.scatter(X,y)
plt.plot(X_train,lr.predict(X_train))
plt.xlabel('X')
plt.ylabel('y')

poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_transformed = poly.fit_transform(X_train)
X_test_transformed = poly.transform(X_test)

lr_new = LinearRegression()
lr_new.fit(X_train_transformed,y_train)
y_pred_transformed = lr_new.predict(X_test_transformed)
r2_score(y_test,y_pred_transformed)

X_new = np.linspace(-3,3,200).reshape(200,1)
X_new_poly = poly.transform(X_new)
y_new_pred = lr_new.predict(X_new_poly)

plt.plot(X_train,y_train,'r.',label = "Training Points")
plt.plot(X_test,y_test,'g.',label = "Testing Points")
plt.plot(X_new,y_new_pred,'b-',label = "Predictions")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()