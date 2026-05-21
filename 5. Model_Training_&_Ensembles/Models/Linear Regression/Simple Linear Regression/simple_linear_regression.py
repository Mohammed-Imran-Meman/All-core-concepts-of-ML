import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class myLR:
  def __init__(self):
    self.m = None
    self.b = None

  def fit(self,X_train, y_train):
    num = 0
    den = 0
    for i in range(X_train.shape[0]):
      num = num + ((X_train.iloc[i] - X_train.mean()) * (y_train.iloc[i] - y_train.mean()))
      den = den + np.square(X_train.iloc[i]-X_train.mean())

    self.m = num/den
    self.b = y_train.mean() - self.m*X_train.mean()

  def predict(self,X_test):
    return self.m * X_test + self.b
  
  def coef_(self):
    return self.m
  def intercept_(self):
    return self.b 

df = pd.read_csv('placement.csv')
df.head()

X = df.iloc[:,0]
y = df['package']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = myLR()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_score(y_test, y_pred)

lr.coef_()
lr.intercept_()