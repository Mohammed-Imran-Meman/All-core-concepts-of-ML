import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

df = pd.read_csv('customer.csv')
df.sample(5) 

df = df.iloc[:,2:]
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:2],df['purchased'],test_size=0.2,random_state=0)

oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])

X_train_encoded = oe.fit_transform(X_train)
X_test_encoded = oe.transform(X_test)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)