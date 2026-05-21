import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X,y = make_regression(n_samples=100,n_features=2,n_informative=2,n_targets=1,noise=50)
df = pd.DataFrame({'Feature1':X[:,0],'Feature2':X[:,1],'Target':y})
df.head()
df.shape

px.scatter_3d(x=df['Feature1'], y=df['Feature2'], z=df['Target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))

x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)

xGrid, yGrid = np.meshgrid(x, y)

final = np.vstack((xGrid.ravel().reshape(1,100),yGrid.ravel().reshape(1,100))).T

z = lr.predict(final).reshape(10, 10)

fig = px.scatter_3d(df, x='Feature1', y='Feature2', z='Target')

fig.add_trace(go.Surface(x = x, y = y, z =z ))