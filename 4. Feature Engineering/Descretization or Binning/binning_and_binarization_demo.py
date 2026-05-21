import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
df.dropna(inplace=True)
df.head()

# Descretization/Binning

X = df.iloc[:,1:]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)

print(np.mean(cross_val_score(clf,X,y,cv=10,scoring='accuracy')))

def binning(bins, strategy):

  Kbin = KBinsDiscretizer(n_bins=bins,strategy=strategy, encode='ordinal')

  trf = ColumnTransformer([
    ('first', Kbin,['Age']),
    ('second', Kbin, ['Fare'])
  ])

  X_trf = trf.fit_transform(X)
  output = pd.DataFrame({
    'Age':X['Age'],
    'Age_trf':X_trf[:,0],
    'Fare':X['Fare'],
    'Fare_trf':X_trf[:,1]
  }) 
  output['Age_Labels'] = pd.cut(x=X['Age'],bins=trf.named_transformers_['first'].bin_edges_[0].tolist())
  output['Fare_Labels'] = pd.cut(x=X['Fare'],bins=trf.named_transformers_['second'].bin_edges_[0].tolist())

  output.sample(5)
  clf = DecisionTreeClassifier()
  print(float(np.mean(cross_val_score(clf,X_trf,y,cv=10,scoring='accuracy'))))

  plt.figure(figsize=(14,4))
  plt.subplot(121)
  plt.hist(X['Age'])
  plt.title("Before")

  plt.subplot(122)
  plt.hist(X_trf[:,0],color='red')
  plt.title("After")

  plt.show()
    
  plt.figure(figsize=(14,4))
  plt.subplot(121)
  plt.hist(X['Fare'])
  plt.title("Before")

  plt.subplot(122)
  plt.hist(X_trf[:,1],color='red')
  plt.title("Fare")

  plt.show()


binning(5,'uniform')
binning(5,'quantile')
binning(5,'kmeans')



# Binarization

df2 = pd.read_csv('train.csv')[['Age','Fare','SibSp','Parch','Survived']]
df2.dropna(inplace=True)
df2.head()
df2['Family'] = df2['SibSp'] + df2['Parch']
df2.drop(columns=['SibSp','Parch'])

X = df2.drop(columns='Survived')
y = df2['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)

print(np.mean(cross_val_score(clf,X,y,cv=10,scoring='accuracy')))

trf = ColumnTransformer([
    ('bin',Binarizer(copy=False),['Family'])
],remainder='passthrough')

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

clf = DecisionTreeClassifier()
clf.fit(X_train_trf,y_train)
y_pred2 = clf.predict(X_test_trf)
accuracy_score(y_test,y_pred2)

X_trf = trf.fit_transform(X)
print(np.mean(cross_val_score(clf,X_trf,y,cv=10,scoring='accuracy')))