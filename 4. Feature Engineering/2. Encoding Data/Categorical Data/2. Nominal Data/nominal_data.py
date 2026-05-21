import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('cars.csv')
df.shape
df['brand'].value_counts()
df['brand'].nunique()

df['owner'].value_counts()
df['owner'].nunique()
df['fuel'].nunique()

# OHE using Pandas, it is not used generally as it can randomly generate sequence of columns
pd.get_dummies(df, columns=['fuel','owner'],drop_first=True)

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:4],df.iloc[:,-1],test_size=0.3)

ohe = OneHotEncoder(drop='first', sparse_output=False)

X_train_encoded = ohe.fit_transform(X_train[['fuel','owner']])
X_test_encoded = ohe.transform(X_test[['fuel','owner']])

np.hstack((X_train[['brand','km_driven']].values,X_train_encoded))

counts = df['brand'].value_counts()
threshold = 100

repl = counts[counts<=threshold].index

pd.get_dummies(df['brand'].replace(repl, 'uncommen'),dtype=int)