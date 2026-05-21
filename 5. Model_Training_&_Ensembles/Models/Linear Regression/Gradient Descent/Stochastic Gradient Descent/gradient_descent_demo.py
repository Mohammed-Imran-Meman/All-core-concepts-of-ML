import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X,y = make_regression(n_samples=100,n_features=15,n_informative=1,n_targets=1,noise=20)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)

lr.coef_
lr.intercept_

class SGDRegression:
  def __init__(self,learning_rate=0.01,epoch=100):
    # m is given
    self.intercept_ = None
    self.coef_ = None
    self.lr = learning_rate
    self.epoch = epoch

  def fit(self,X_train,y_train):
    self.intercept_ = 0
    self.coef_ = np.ones(X_train.shape[1])
    for i in range(self.epoch):
      for j in range(X_train.shape[0]):
        idx = np.random.randint(0,X_train.shape[0])
        # code for finding intercept
        y_hat = self.intercept_ + np.dot(X_train[idx],self.coef_) 
        intercept_dr = -2 * (y_train[idx]-y_hat)
        self.intercept_ = self.intercept_ - (self.lr * intercept_dr)

        # code for finding coef
        coef_dr = -2 * np.dot((y_train[idx]-y_hat),X_train[idx])
        self.coef_ = self.coef_ - (self.lr * coef_dr)

  def predict(self,X_test):
    return np.dot(X_test,self.coef_) + self.intercept_

gd = SGDRegression(0.01,100)
gd.fit(X_train,y_train)
y_pred = gd.predict(X_test)
r2_score(y_test,y_pred)

gd.coef_
gd.intercept_