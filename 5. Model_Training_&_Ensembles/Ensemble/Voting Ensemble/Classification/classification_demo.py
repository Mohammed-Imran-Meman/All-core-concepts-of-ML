import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

df = pd.read_csv('iris.csv')
df.head()

df = df.iloc[:,1:]
df.head()

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

sns.pairplot(df,hue='Species')
new_df = df[df['Species']!=0][['SepalLengthCm','SepalWidthCm','Species']]
new_df.head()

X = df.iloc[:,0:2]
y = df['Species']

clf1 = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = RandomForestClassifier()

estimators = [('lr',clf1),('knn',clf2),('rf',clf3)]

for estimator in estimators:
  x = cross_val_score(estimator[1],X,y,scoring='accuracy')
  print(estimator[0],np.round(np.mean(x),2))

# Hard Voting
vc = VotingClassifier(estimators=estimators,voting='hard')
x = cross_val_score(vc,X,y,scoring='accuracy')
print(np.round(np.mean(x),2))

# Soft Voting
vc = VotingClassifier(estimators=estimators,voting='soft')
x = cross_val_score(vc,X,y,scoring='accuracy')
print(np.round(np.mean(x),2))

# Weighted Voting
vc = VotingClassifier(estimators=estimators,voting='soft',weights=[3,1,1])
x = cross_val_score(vc,X,y,scoring='accuracy')
print(np.round(np.mean(x),2))

for i in range(1,4):
  for j in range(1,4):
    for k in range(1,4):
      vc = VotingClassifier(estimators=estimators,voting='soft',weights=[i,j,k])
      x = cross_val_score(vc,X,y,scoring='accuracy')
      print('for i = {}, j = {}, k = {}'.format(i,j,k),np.round(np.mean(x),2))

# Classifiers of Same Algorithms
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)

svm1 = SVC(probability=True, kernel='poly', degree=1)
svm2 = SVC(probability=True, kernel='poly', degree=2)
svm3 = SVC(probability=True, kernel='poly', degree=3)
svm4 = SVC(probability=True, kernel='poly', degree=4)
svm5 = SVC(probability=True, kernel='poly', degree=5)

estimators = [('svm1',svm1),('svm2',svm2),('svm3',svm3),('svm4',svm4),('svm5',svm5)]

for estimator in estimators:
    x = cross_val_score(estimator[1],X,y,cv=10,scoring='accuracy')
    print(estimator[0],np.round(np.mean(x),2))

vc1 = VotingClassifier(estimators=estimators,voting='soft')
x = cross_val_score(vc1,X,y,scoring='accuracy')
print(np.round(np.mean(x),2))