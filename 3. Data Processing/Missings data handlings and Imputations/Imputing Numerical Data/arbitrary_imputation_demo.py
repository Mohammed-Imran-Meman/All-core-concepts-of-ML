import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('titanic_toy.csv')
df.head()
df.isnull().mean()

X = df.iloc[:,:3]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.isnull().mean()

X_train['Age_99'] = X_train['Age'].fillna(99)
X_train['Age_-1'] = X_train['Age'].fillna(-1)
X_train['Fare_999'] = X_train['Fare'].fillna(999)
X_train['Fare_-1'] = X_train['Fare'].fillna(-1)

X_train.cov()
X_train.corr()

X_train['Age'].var()
X_train['Age_-1'].var()
X_train['Age_99'].var()
X_train['Fare'].var()
X_train['Fare_-1'].var()
X_train['Fare_999'].var()


plt.figure(figsize=(7,7))
# plt.subplot(131)
sns.kdeplot(X_train['Age'])
# plt.subplot(132)
sns.kdeplot(X_train['Age_99'],color='red')
# plt.subplot(133)
sns.kdeplot(X_train['Age_-1'],color='green')


plt.figure(figsize=(14,7))
plt.subplot(131)
sns.kdeplot(X_train['Fare'])
plt.subplot(132)
sns.kdeplot(X_train['Fare_999'],color='red')
plt.subplot(133)
sns.kdeplot(X_train['Fare_-1'],color='green')


# mean-median by using ScikitLearn


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trf = ColumnTransformer([
  ('age_fare_original', 'passthrough', ['Age', 'Fare']),
  ('age_fare_99', SimpleImputer(strategy='constant',fill_value=(99)), ['Age','Fare']),
  ('age_fare_-1', SimpleImputer(strategy='constant',fill_value=(-1)), ['Age','Fare'])],remainder='passthrough'
)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
X_train_transformed.shape

pd.DataFrame(X_train_transformed,columns=['Age','Fare','Age_99','Fare_99','Age_-1','Fare_-1','Family'])
pd.DataFrame(X_test_transformed,columns=['Age','Fare','Age_99','Fare_99','Age_-1','Fare_-1','Family'])
trf.named_transformers_['age_fare_99'].statistics_