import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

column_names = ['X', 'Y', 'class']
df = pd.read_csv('ushape.csv', names=column_names)
df.head()

X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values
plt.scatter(X[:,0],X[:,1],c=y)

clf = LogisticRegression()
clf.fit(X,y)

# plotting
plot_decision_regions(X, y.astype('int'), clf, legend=2)

np.mean(cross_val_score(clf,X,y,scoring='accuracy',cv=10))

poly = PolynomialFeatures(degree=3,include_bias=False)
X_trf = poly.fit_transform(X)
clf1 = LogisticRegression()
np.mean(cross_val_score(clf1,X_trf,y,scoring='accuracy',cv=10))

#  All in one function

def plot_decision_boundary(X,y,degree=1): 
  poly = PolynomialFeatures(degree=degree)
  X_trf = poly.fit_transform(X)
  
  clf = LogisticRegression()
  clf.fit(X_trf,y)
  
  accuracy = np.mean(cross_val_score(clf,X_trf,y,scoring='accuracy',cv=10))
  
  a=np.arange(start=X[:,0].min()-1, stop=X[:,0].max()+1, step=0.01)
  b=np.arange(start=X[:,1].min()-1, stop=X[:,1].max()+1, step=0.01)


  XX,YY=np.meshgrid(a,b)
    
  input_array=np.array([XX.ravel(),YY.ravel()]).T

  labels=clf.predict(poly.transform(input_array))
    
  plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.5)
  plt.scatter(X[:,0],X[:,1], c=y)
  plt.title('Degree = {}, accuracy is {}'.format(degree,np.round(accuracy,4)))


plot_decision_boundary(X,y)
plot_decision_boundary(X,y,degree=2)