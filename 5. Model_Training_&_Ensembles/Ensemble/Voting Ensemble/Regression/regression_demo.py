import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data
y = data.target
X = X[:1000]
y = y[:1000]

reg1 = LinearRegression()
reg2 = DecisionTreeRegressor()
reg3 = SVR()

estimators = [('lr',reg1),('dt',reg2),('svr',reg3)]

for estimator in estimators:
  x = cross_val_score(estimator[1],X,y,scoring='r2')
  print(estimator[0],np.round(np.mean(x),2))

vr = VotingRegressor(estimators=estimators)
x = cross_val_score(vr,X,y,scoring='r2')
print(np.round(np.mean(x),2))

# Weighted Voting
vr = VotingRegressor(estimators=estimators,weights=[3,1,1])
x = cross_val_score(vr,X,y,scoring='r2')
print(np.round(np.mean(x),2))

for i in range(1,4):
  for j in range(1,4):
    for k in range(1,4):
      vr = VotingRegressor(estimators=estimators,weights=[i,j,k])
      x = cross_val_score(vr,X,y,scoring='r2')
      print('for i = {}, j = {}, k = {}'.format(i,j,k),np.round(np.mean(x),2))

# Regression of Same Algorithms
dt1 = DecisionTreeRegressor(max_depth=1)
dt2 = DecisionTreeRegressor(max_depth=3)
dt3 = DecisionTreeRegressor(max_depth=5)
dt4 = DecisionTreeRegressor(max_depth=7)
dt5 = DecisionTreeRegressor(max_depth=None)

estimators = [('dt1',dt1),('dt2',dt2),('dt3',dt3),('dt4',dt4),('dt5',dt5)]

for estimator in estimators:
  scores = cross_val_score(estimator[1],X,y,scoring='r2',cv=10)
  print(estimator[0],np.round(np.mean(scores),2))

vr = VotingRegressor(estimators)
scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
print("Voting Regressor",np.round(np.mean(scores),2))