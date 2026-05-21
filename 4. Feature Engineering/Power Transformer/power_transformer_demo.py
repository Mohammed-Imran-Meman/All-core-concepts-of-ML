import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('concrete_data.csv')
df.head()
df.isnull().sum()
df.describe()

X = df.iloc[:,:-1]
y = df['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

r2_score(y_test,y_pred)

lr2 = LinearRegression()
print(np.mean(cross_val_score(lr2,X,y,scoring='r2'))) #by default cv=5

for col in X_train.columns:
  plt.figure(figsize=(14,4))

  plt.subplot(121)
  sns.distplot(X_train[col])
  plt.title(col)

  plt.subplot(122)
  stats.probplot(X_train[col], dist="norm", plot=plt)
  plt.title(col)


# Power Transformer: Box-Cox
pt = PowerTransformer(method='box-cox')

X_train_trans = pt.fit_transform(X_train+0.000001)
X_test_trans = pt.transform(X_test+0.000001)
pd.DataFrame({
  'columns':X_train.columns,
  'calculated_lamda': pt.lambdas_
})

lr3 = LinearRegression()
lr3.fit(X_train_trans,y_train)

y_predict = lr3.predict(X_test_trans)
r2_score(y_test, y_predict)

pt2 = PowerTransformer(method='box-cox')
X_transformed = pt2.fit_transform(X+0.000001)

lr4 = LinearRegression()
print(np.mean(cross_val_score(lr4,X_transformed,y,scoring='r2',cv=10)))

X_transformed = pd.DataFrame(X_transformed,columns=X_train.columns)
for col in X_transformed.columns:
  plt.figure(figsize=(14,4))

  plt.subplot(121)
  stats.probplot(X_train[col], dist="norm", plot=plt)
  plt.title(col)

  plt.subplot(122)
  stats.probplot(X_transformed[col], dist="norm", plot=plt)
  plt.title(col)


# Power Transformer: Yeo-Johnson
pt3 = PowerTransformer() #default method is yeo-johnson

X_train_transform = pt3.fit_transform(X_train)
X_test_transform = pt3.transform(X_test)
pd.DataFrame({
  'columns':X_train.columns,
  'calculated_lamda': pt3.lambdas_
})

lr5 = LinearRegression()
lr5.fit(X_train_transform,y_train)

y_prediction = lr5.predict(X_test_transform)
r2_score(y_test, y_prediction)

pt4 = PowerTransformer()
X_transformed2 = pt4.fit_transform(X)

lr6 = LinearRegression()
print(np.mean(cross_val_score(lr6,X_transformed2,y,scoring='r2',cv=10)))

X_transformed2 = pd.DataFrame(X_transformed2,columns=X_train.columns)
for col in X_transformed2.columns:
  plt.figure(figsize=(14,4))

  plt.subplot(121)
  stats.probplot(X_train[col], dist="norm", plot=plt)
  plt.title(col)

  plt.subplot(122)
  stats.probplot(X_transformed2[col], dist="norm", plot=plt)
  plt.title(col)