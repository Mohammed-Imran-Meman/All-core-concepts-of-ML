import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('train.csv')
df.head()

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:],df['Survived'],test_size=0.2,random_state=42)
X_train.head()

trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')

trf2 = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')

trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])

trf4 = SelectKBest(score_func=chi2,k=8)

trf5 = DecisionTreeClassifier()

pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
accuracy_score(y_test,y_pred)

pickle.dump(pipe,open('models/pipe.pkl','wb'))