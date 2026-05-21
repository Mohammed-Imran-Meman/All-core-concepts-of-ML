import numpy as np
class MyRidge:
  def __init__(self,alpha=0.1):
    self.alpha = alpha
    self.intercept_ = None
    self.coef_ = None

  def  fit(self, X_train,y_train):
    X_train = np.insert(X_train,0,1,axis=1)
    I = np.identity(X_train.shape[1])
    I[0][0] = 0
    w = np.linalg.inv(np.dot(X_train.T,X_train) + self.alpha * I).dot(X_train.T).dot(y_train)
    self.intercept_ = w[0]
    self.coef_ = w[1:]

  def predict(self,X_test):
    return np.dot(X_test,self.coef_) + self.intercept_
  