import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

df1 = pd.read_csv('data/heart.csv')
df1.head()

X1 = df1.iloc[:,:-1]
y1 = df1['output']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf1.fit(X1_train,y1_train)
clf2.fit(X1_train,y1_train)

y1_pred1 = clf1.predict(X1_test)
y1_pred2 = clf2.predict(X1_test)

print("Accuracy of LR",accuracy_score(y1_test,y1_pred1))
print("Accuracy of DT",accuracy_score(y1_test,y1_pred2))

print("Confusion Matrix of LR\n")
pd.DataFrame(confusion_matrix(y1_test,y1_pred1),columns=list(range(0,2)))
print("Confusion Matrix of DT\n")
pd.DataFrame(confusion_matrix(y1_test,y1_pred2),columns=list(range(0,2)))

result = pd.DataFrame()
result['Actual Label'] = y1_test
result['Logistic Regression Prediction'] = y1_pred1
result['Decision Tree Prediction'] = y1_pred2

# For multiple outputs
df2 = pd.read_csv('data/train.csv')

X2_train,X2_test,y2_train,y2_test = train_test_split(df2.iloc[:,1:],df2.iloc[:,0],test_size=0.2,random_state=2)

clf3 = LogisticRegression()
clf4 = DecisionTreeClassifier()
clf3.fit(X2_train,y2_train)
clf4.fit(X2_train,y2_train)

y2_pred1 = clf3.predict(X2_test)
y2_pred2 = clf4.predict(X2_test)

print("Accuracy of LR",accuracy_score(y2_test,y2_pred1))
print("Accuracy of DT",accuracy_score(y2_test,y2_pred2))

print("Confusion Matrix of LR\n")
pd.DataFrame(confusion_matrix(y2_test,y2_pred1),columns=list(range(0,10)))
print("Confusion Matrix of DT\n")
pd.DataFrame(confusion_matrix(y2_test,y2_pred2),columns=list(range(0,10)))
# Avg = None
precision_score(y2_test,y2_pred1,average=None)

recall_score(y2_test,y2_pred1,average=None)

f1_score(y2_test,y2_pred1,average=None)

precision_score(y2_test,y2_pred1,average='macro')

recall_score(y2_test,y2_pred1,average='macro')

f1_score(y2_test,y2_pred1,average='macro')

precision_score(y2_test,y2_pred1,average='weighted')

recall_score(y2_test,y2_pred1,average='weighted')

f1_score(y2_test,y2_pred1,average='weighted')

print(classification_report(y2_test,y2_pred1))