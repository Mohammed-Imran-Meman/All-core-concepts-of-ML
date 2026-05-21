import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('covid_toy.csv')
df.head()
df.sample(10)
df.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:5],df['has_covid'],test_size=0.2)

# Without column transformer
si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])
X_test_fever = si.transform(X_test[['fever']])
X_train_fever.shape

oe = OrdinalEncoder(categories=[['Mild','Strong']])
X_train_cough = oe.fit_transform(X_train[['cough']])
X_test_cough = oe.transform(X_test[['cough']])
X_train_cough.shape

ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_gender_city = ohe.fit_transform(X_train[['gender','city']])
X_test_gender_city = ohe.transform(X_test[['gender','city']])
X_train_gender_city.shape

X_train_age = X_train.drop(columns=['gender','fever','cough','city']).values
X_test_age = X_test.drop(columns=['gender','fever','cough','city']).values
X_train_age.shape

X_train_transformed = np.concatenate((X_train_age,X_train_fever,X_train_gender_city, X_train_cough),axis=1)
X_test_transformed = np.concatenate((X_test_age,X_test_fever,X_test_gender_city, X_test_cough),axis=1)
X_train_transformed.shape

# With column transformer
transformer = ColumnTransformer(transformers=[('tnf1',SimpleImputer(),['fever']),('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),('tnf3',OneHotEncoder(drop='first',sparse_output=False),['gender','city'])],remainder='passthrough')

X_train_transformed2 = transformer.fit_transform(X_train)
X_test_transformed2 = transformer.transform(X_test)
X_train_transformed2.shape