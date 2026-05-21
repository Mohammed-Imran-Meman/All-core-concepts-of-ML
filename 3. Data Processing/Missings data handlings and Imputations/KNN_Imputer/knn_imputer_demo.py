import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv')[['Age','Pclass','Fare','Survived']]
df.head()
df.isnull().mean() * 100

X = df.iloc[:,:3]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# imputing using Simple Imputer
si = SimpleImputer()

X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_trf,y_train)
y_pred = lr.predict(X_test_trf)

accuracy_score(y_test,y_pred)


# imputing using KNN Imputer
knn = KNNImputer(n_neighbors=10,weights='uniform')

X_train_trf2 = knn.fit_transform(X_train)
X_test_trf2 = knn.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_trf2,y_train)
y_pred = lr.predict(X_test_trf2)

accuracy_score(y_test,y_pred)
