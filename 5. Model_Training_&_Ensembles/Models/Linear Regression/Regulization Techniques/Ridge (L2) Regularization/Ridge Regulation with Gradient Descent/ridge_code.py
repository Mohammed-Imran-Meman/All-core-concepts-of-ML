import numpy as np
class MyRidge:
  def __init__(self,epochs,learning_rate,alpha): 
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.alpha = alpha
    self.coef_ = None
    self.intercept_ = None
        
  def fit(self,X_train,y_train):
        
    self.coef_ = np.ones(X_train.shape[1])
    self.intercept_ = 0
    theta = np.insert(self.coef_,0,self.intercept_)
        
    X_train = np.insert(X_train,0,1,axis=1)
        
    for i in range(self.epochs):
      theta_der = np.dot(X_train.T,X_train).dot(theta) - np.dot(X_train.T,y_train) + self.alpha*theta
      theta = theta - self.learning_rate*theta_der
        
    self.coef_ = theta[1:]
    self.intercept_ = theta[0]
    
  def predict(self,X_test):
        
    return np.dot(X_test,self.coef_) + self.intercept_