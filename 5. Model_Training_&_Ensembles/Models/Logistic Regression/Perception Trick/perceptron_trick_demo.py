import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=30)

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)

def perceptron(X,y):
  X = np.insert(X,0,1,axis=1)
  weights = np.ones(X.shape[1])
  lr = 0.1

  for i in range(1000):
    j = np.random.randint(0,X.shape[0])
    y_hat = int(np.dot(X[j],weights) > 0)
    weights = weights + lr * (y[j] - y_hat)*X[j]

  return weights[0],weights[1:]

intercept_,coef_ = perceptron(X,y)

# m = -A/C  and b = -B/C
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])

x_input = np.linspace(-3,3,100)
y_input = m*x_input + b

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)

# to improve our output we'll use sigmoid func
def sigmoid(z):
  return 1/(1+np.exp(-z))

# func is same just using sigmoid rather then 0 and 1 output of y_hat
def perceptron_(X,y):
  X = np.insert(X,0,1,axis=1)
  weights = np.ones(X.shape[1])
  lr = 0.1

  for i in range(1000):
    j = np.random.randint(0,X.shape[0])
    y_hat = sigmoid(np.dot(X[j],weights))
    weights = weights + lr * (y[j] - y_hat)*X[j]

  return weights[0],weights[1:]

intercept_,coef_ = perceptron_(X,y)

m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])

x_input = np.linspace(-3,3,100)
y_input = m*x_input + b

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)