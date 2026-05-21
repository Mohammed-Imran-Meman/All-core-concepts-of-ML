import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from ridge_code_of_multiple_regression import MyRidge

X,y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("RMSE:",root_mean_squared_error(y_test,y_pred))

r = Ridge(alpha=0.1)
r.fit(X_train,y_train)
y_pred = r.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("RMSE:",root_mean_squared_error(y_test,y_pred))
r.coef_

# getting Ridge model from the file of the code of multiple linear regression

my_ridge = MyRidge()
my_ridge.fit(X_train,y_train)
y_pred_own = my_ridge.predict(X_test)
r2_score(y_test,y_pred_own)
my_ridge.coef_


m = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1 ** 2 - 2 * x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2)

def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)

alphas = [0, 20, 200]
cs = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Datapoints')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    # Plot
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))


plt.legend()