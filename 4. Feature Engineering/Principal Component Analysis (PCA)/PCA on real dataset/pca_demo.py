import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


df = pd.read_csv('train.csv')
df.sample()
plt.imshow(df.iloc[33318, 1:].values.reshape(28, 28))

X = df.iloc[:,1:]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

start = time.time()
y_pred = knn.predict(X_test)
print(time.time()-start)
accuracy_score(y_test,y_pred)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=None)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train_trf,y_train)

y_pred = knn.predict(X_test_trf)
accuracy_score(y_test,y_pred)

# accuracy for single to all features
for i in range(1,785):
  pca = PCA(n_components=i)
  X_train_trf = pca.fit_transform(X_train)
  X_test_trf = pca.transform(X_test)

  knn = KNeighborsClassifier()
  knn.fit(X_train_trf,y_train)

  y_pred = knn.predict(X_test_trf)
  print(accuracy_score(y_test,y_pred))

# plotting 2D visualisation
pca = PCA(n_components=2)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

y_train_trf = y_train.astype(str)
fig = px.scatter(x=X_train_trf[:,0],
                    y=X_train_trf[:,1],
                    color=y_train_trf,
                    color_discrete_sequence=px.colors.qualitative.G10)
fig.show()

# plotting 3D visualisation
pca = PCA(n_components=3)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

y_train_trf = y_train.astype(str)
fig = px.scatter_3d(x=X_train_trf[:,0],
                    y=X_train_trf[:,1],
                    z=X_train_trf[:,2],
                    color=y_train_trf,
                    color_discrete_sequence=px.colors.qualitative.G10)
fig.show()

# Eigen values
pca.explained_variance_

# Eigen vectors
pca.components_

# how much Eigen Values takes place in the whole data
pca.explained_variance_ratio_

# add the ratios 
np.cumsum(pca.explained_variance_ratio_)

# plotting this data
plt.plot(np.cumsum(pca.explained_variance_ratio_))

# for whole data
pca = PCA(n_components=None)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

