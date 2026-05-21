import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import MissingIndicator,SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
df.head()

X = df.iloc[:,1:]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without using missing indicator
si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_trf,y_train)
y_pred = lr.predict(X_test_trf)
accuracy_score(y_test,y_pred)


# With using missing indicator
mi = MissingIndicator()
X_train_missing = mi.fit_transform(X_train)
X_test_missing = mi.transform(X_test)

X_train['Age_Missing'] = X_train_missing
X_test['Age_Missing'] = X_test_missing

si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_trf,y_train)
y_pred = lr.predict(X_test_trf)
accuracy_score(y_test,y_pred)



# With using missing indicator (inbuilt in Simple imputer)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


si = SimpleImputer(add_indicator=True)
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_trf,y_train)
y_pred = lr.predict(X_test_trf)
accuracy_score(y_test,y_pred)