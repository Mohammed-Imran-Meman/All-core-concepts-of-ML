import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.csv')
df.head()

df = df.iloc[:,1:]
df.head()

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

df = df[df['Species']!=0][['SepalWidthCm','PetalLengthCm','Species']]
new_df.head()
plt.scatter(df['SepalWidthCm'],df['PetalLengthCm'],c=df['Species'],cmap='winter')

df.shape
df = df.sample(100)
df_train = df.iloc[:60,:].sample(10)
df_val = df.iloc[60:80,:].sample(5)
df_test = df.iloc[80:,:].sample(5)

X_test = df_val.iloc[:,0:2].values
y_test = df_val.iloc[:,-1].values

def evaluate(clf,X,y):
  clf.fit(X,y)
  plot_tree(clf)
  plt.show()
  plot_decision_regions(X.values, y.values, clf=clf, legend=2)
  y_pred = clf.predict(X_test)
  print(accuracy_score(y_test,y_pred))

# Bagging starts from here
# Data for Tree 1
df_bag = df_train.sample(8,replace=True)

X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

dt_bag1 = DecisionTreeClassifier()
evaluate(dt_bag1,X,y)

# Data for Tree 2
df_bag = df_train.sample(8,replace=True)

X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

dt_bag2 = DecisionTreeClassifier()
evaluate(dt_bag1,X,y)

# Data for Tree 3
df_bag = df_train.sample(8,replace=True)

X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

dt_bag3 = DecisionTreeClassifier()
evaluate(dt_bag1,X,y)

# Predition 
df_test
print("Predictor 1",dt_bag1.predict(np.array([2.8,4.1]).reshape(1,2)))
print("Predictor 2",dt_bag2.predict(np.array([2.8,4.1]).reshape(1,2)))
print("Predictor 3",dt_bag3.predict(np.array([2.8,4.1]).reshape(1,2)))
