import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Construction
df = pd.read_csv('train.csv')[['Age','Pclass','SibSp','Parch','Survived']]
df.head()

df.dropna(inplace=True)

X = df.iloc[:,:-1]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

float(np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=10)))

X['Family_size'] = X['SibSp'] + X['Parch'] + 1
X.head()

def myfunc(num):
  if num == 1:
    return 0
  elif num > 1 & num <= 4:
    return 1
  else:
    return 2
  
X['Family_type'] = X['Family_size'].apply(myfunc)
X.head()

X.drop(columns=['SibSp','Parch','Family_size'],inplace=True)

float(np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=10)))

# Splitting
df = pd.read_csv('train.csv')
df.head()
df['Title'] = df['Name'].str.split(',', expand=True)[1].str.split('.',expand=True)[0]
(df.groupby('Title')['Survived'].mean()).sort_values(ascending=False)

df['Is_Married'] = 0
df.loc[df['Title'].str.strip() == 'Mrs', 'Is_Married'] = 1
df['Is_Married'].value_counts()