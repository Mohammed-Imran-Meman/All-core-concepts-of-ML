import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data.csv')
df.head()
df.drop(columns = ['id','Unnamed: 32'] , inplace = True)
X = df.iloc[:,1:]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
accuracy_score(y_test,y_pred)

# getting best accuracy by setting n neighbiors
scores = []

for i in range(1,16):    
  knn = KNeighborsClassifier(n_neighbors=i)    
  knn.fit(X_train,y_train)    
  y_pred = knn.predict(X_test)
  scores.append(accuracy_score(y_test, y_pred))

# plotting
plt.plot(range(1,16),scores)