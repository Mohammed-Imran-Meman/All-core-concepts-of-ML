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

mean_age = X_train['Age'].mean()
median_age = X_train['Age'].median()

mean_fare = X_train['Fare'].mean()
median_fare = X_train['Fare'].median()

X_train['Age_mean'] = X_train['Age'].fillna(mean_age)
X_train['Age_median'] = X_train['Age'].fillna(median_age)
X_train['Fare_mean'] = X_train['Fare'].fillna(mean_fare)
X_train['Fare_median'] = X_train['Fare'].fillna(median_fare)

X_train.cov()
X_train.corr()

X_train['Age'].var()
X_train['Age_median'].var()
X_train['Age_mean'].var()
X_train['Fare'].var()
X_train['Fare_median'].var()
X_train['Fare_mean'].var()


plt.figure(figsize=(7,7))
# plt.subplot(131)
sns.kdeplot(X_train['Age'])
# plt.subplot(132)
sns.kdeplot(X_train['Age_mean'],color='red')
# plt.subplot(133)
sns.kdeplot(X_train['Age_median'],color='green')


plt.figure(figsize=(14,7))
plt.subplot(131)
sns.kdeplot(X_train['Fare'])
plt.subplot(132)
sns.kdeplot(X_train['Fare_mean'],color='red')
plt.subplot(133)
sns.kdeplot(X_train['Fare_median'],color='green')


# mean-median by using ScikitLearn


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trf = ColumnTransformer([
  ('age_fare_original', 'passthrough', ['Age', 'Fare']),
  ('age_fare_mean', SimpleImputer(strategy='mean'), ['Age','Fare']),
  ('age_fare_median', SimpleImputer(strategy='median'), ['Age','Fare'])],remainder='passthrough'
)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
X_train_transformed.shape

pd.DataFrame(X_train_transformed,columns=['Age','Fare','Age_mean','Fare_mean','Age_median','Fare_median','Family'])
pd.DataFrame(X_test_transformed,columns=['Age','Fare','Age_mean','Fare_mean','Age_median','Fare_median','Family'])
trf.named_transformers_['age_fare_mean'].statistics_