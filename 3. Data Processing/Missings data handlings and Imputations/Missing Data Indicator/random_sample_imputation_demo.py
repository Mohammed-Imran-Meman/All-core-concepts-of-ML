import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
df.head()

X = df.iloc[:,1:]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['Age_imputed'] = X_train['Age']
X_test['Age_imputed'] = X_test['Age']

X_train['Age_imputed'][X_train['Age_imputed'].isnull()] = X_train['Age'].dropna().sample(X_train['Age'].isnull().sum()).values
X_test['Age_imputed'][X_test['Age_imputed'].isnull()] = X_test['Age'].dropna().sample(X_test['Age'].isnull().sum()).values

sns.kdeplot(X_train['Age'],label='Original Age',linestyle = '--')
sns.kdeplot(X_train['Age_imputed'],label='Imputed Age',linestyle = '-.')

X_train[['Age', 'Age_imputed']].boxplot()
