import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X,y = make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=20)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)

lr.coef_
lr.intercept_

class GDRegression:
  def __init__(self,learning_rate,epoch):
    # m is given
    self.m = 43.79
    self.b = 100
    self.lr = learning_rate
    self.epoch = epoch

  def fit(self,X,y):
    for i in range(self.epoch):
      loss_slope = -2 * np.sum(y - self.m*X.ravel()-self.b)
      self.b = self.b - self.lr*loss_slope
    print(self.b)

gd = GDRegression(0.001,100)
gd.fit(X,y)


class GDRegression:
  def __init__(self,learning_rate,epoch):
    # m and b both is not given, we will predict it...
    self.m = 100
    self.b = 100
    self.lr = learning_rate
    self.epoch = epoch

  def fit(self,X,y):
    for i in range(self.epoch):
      loss_slope_b = -2 * np.sum(y - self.m*X.ravel()-self.b)
      loss_slope_m = -2 * np.sum((y - self.m*X.ravel()-self.b)*X.ravel())
      self.b = self.b - self.lr*loss_slope_b
      self.m = self.m - self.lr*loss_slope_m
    print(self.m, self.b)

  def predict(self,X):
    return self.m*X + self.b

gd = GDRegression(0.001,100)
gd.fit(X_train,y_train)
y_pred = gd.predict(X_test)
r2_score(y_test,y_pred)