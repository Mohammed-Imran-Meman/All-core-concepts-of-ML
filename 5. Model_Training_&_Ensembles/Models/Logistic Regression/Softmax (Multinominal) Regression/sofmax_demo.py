import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

df = sns.load_dataset('iris')
df.head()

encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

df = df[['sepal_length','petal_length','species']]

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(multi_class='multinomial')
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

pd.DataFrame(confusion_matrix(y_test,y_pred))

query = np.array([[3.4,2.7]])
clf.predict_proba(query)

clf.predict(query)

# plotting
plot_decision_regions(X.values, y.values, clf, legend=2)

plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.title('Softmax on Iris')

plt.show()