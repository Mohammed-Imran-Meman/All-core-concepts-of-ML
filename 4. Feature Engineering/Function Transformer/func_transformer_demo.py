import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv', usecols=['Age','Fare','Survived'])
df.head()
df.isnull().sum()
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.isnull().sum()
X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

plt.figure(figsize=(14,4))

plt.subplot(121)
sns.distplot(X_train['Age'])
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(X_train['Age'], dist="norm",plot=plt)
plt.title('Age QQ Plot')


plt.figure(figsize=(14,4))

plt.subplot(121)
sns.distplot(X_train['Fare'])
plt.title('Fare PDF')

plt.subplot(122)
stats.probplot(X_train['Fare'], dist="norm",plot=plt)
plt.title('Fare QQ Plot')

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train,y_train)
clf2.fit(X_train,y_train)

y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("Accuracy of LR:",accuracy_score(y_test,y_pred))
print("Accuracy of DT:",accuracy_score(y_test,y_pred2))

# trf = FunctionTransformer(func=np.log1p)
# X_train_transformed = trf.fit_transform(X_train)
# X_test_transformed = trf.transform(X_test)

# clf_transformed = LogisticRegression()
# clf2_transformed = DecisionTreeClassifier()

# clf_transformed.fit(X_train_transformed,y_train)
# clf2_transformed.fit(X_train_transformed,y_train)

# y_pred_transformed = clf_transformed.predict(X_test_transformed)
# y_pred2_transformed = clf2_transformed.predict(X_test_transformed)

# print("Accuracy of LR:",accuracy_score(y_test,y_pred_transformed))
# print("Accuracy of DT:",accuracy_score(y_test,y_pred2_transformed))

# X_transformed = trf.fit_transform(X)
# clf_X_transformed = LogisticRegression()
# clf2_X_transformed = DecisionTreeClassifier()

# print("LR",np.mean(cross_val_score(clf_X_transformed,X_transformed,y,scoring='accuracy',cv=10)))
# print("DT",np.mean(cross_val_score(clf2_X_transformed,X_transformed,y,scoring='accuracy',cv=10)))

# plt.figure(figsize=(14,4))

# plt.subplot(121)
# stats.probplot(X_train['Fare'], dist="norm",plot=plt)
# plt.title('Fare QQ Plot Before')

# plt.subplot(122)
# stats.probplot(X_transformed['Fare'], dist="norm",plot=plt)
# plt.title('Fare QQ Plot After')

# only transforming Fare column because it is skewed right

trf = ColumnTransformer([
  ('log',FunctionTransformer(func=np.log1p),['Fare'])
],remainder='passthrough')

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed,y_train)
clf2.fit(X_train_transformed,y_train)

y_pred = clf.predict(X_test_transformed)
y_pred2 = clf2.predict(X_test_transformed)

print("LR:",accuracy_score(y_test,y_pred))
print("DT:",accuracy_score(y_test,y_pred2))

# defining function for function transformer

def apply_transformer(transform):

  X = df.iloc[:,1:3]
  y = df.iloc[:,0]

  trf = ColumnTransformer([
    ('impute',SimpleImputer(),['Age']),
    ('func_transformer',FunctionTransformer(transform),['Fare'])
    ],remainder='passthrough')

  X_transformed = pd.DataFrame(
    trf.fit_transform(X), columns=['Age','Fare']
  )
  clf = LogisticRegression()

  print("LR:",np.mean(cross_val_score(clf,X_transformed,y,scoring='accuracy',cv=10)))

  plt.figure(figsize=(14,4))

  plt.subplot(121)
  stats.probplot(X['Fare'], dist="norm",plot=plt)
  plt.title('Fare QQ Plot Before')

  plt.subplot(122)
  stats.probplot(X_transformed['Fare'], dist="norm",plot=plt)
  plt.title('Fare QQ Plot After')

apply_transformer(np.log1p)
apply_transformer(np.sin)
apply_transformer(np.cos)
apply_transformer(lambda x: x)
apply_transformer(lambda x: 1/(x+0.00000001))
apply_transformer(lambda x: x**2)
apply_transformer(lambda x: x ** 0.5)