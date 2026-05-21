import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X,y = load_diabetes(return_X_y=True)
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)

class MyLR:  
  def __init__(self):
    self.coef_ = None
    self.intercept_ = None
        
  def fit(self,X_train,y_train):
    X_train = np.insert(X_train,0,1,axis=1)
        
    betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
    self.intercept_ = betas[0]
    self.coef_ = betas[1:]
    
  def predict(self,X_test):
    y_pred = np.dot(X_test,self.coef_) + self.intercept_
    return y_pred
  
lr1 = MyLR()
lr1.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
r2_score(y_test,y_pred1)
lr1.coef_