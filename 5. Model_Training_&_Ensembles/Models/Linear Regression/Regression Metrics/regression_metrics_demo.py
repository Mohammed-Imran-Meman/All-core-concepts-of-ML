import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from adjusted_r2 import adjusted_r2
df = pd.read_csv('placement.csv')
df.head()

plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package')

X = df.iloc[:,0:1]
y = df.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')

print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))
print('RMSE',np.sqrt(mean_squared_error(y_test,y_pred)))
print('r2',r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print('Adjusted_r2',adjusted_r2(r2,X_test))


# Checking Adjusted r2 after adding new non-linear random column
new_df1 = df.copy()
new_df1['random_feature'] = np.random.random(200)

new_df1 = new_df1[['cgpa','random_feature','package']]
new_df1.head()

plt.scatter(new_df1['random_feature'],new_df1['package'])
plt.xlabel('random_feature')
plt.ylabel('Package(in lpa)')

X1 = new_df1.iloc[:,0:2]
y1 = new_df1.iloc[:,-1]
X_train1,X_test1,y_train,y_test = train_test_split(X1,y1,test_size=0.2,random_state=2)
lr1 = LinearRegression()
lr1.fit(X_train1,y_train)
y_pred1 = lr1.predict(X_test1)
new_r1 = r2_score(y_test,y_pred1)

ad1 = adjusted_r2(new_r1, X_test1)


# Checking Adjusted r2 after adding new linear random column
new_df2 = df.copy()
new_df2['iq'] = new_df2['package'] + (np.random.randint(-12,12,200)/10)
new_df2 = new_df2[['cgpa','iq','package']]

plt.scatter(new_df2['iq'],new_df2['package'])
plt.xlabel('iq')
plt.ylabel('Package(in lpa)')


X2 = new_df2.iloc[:,0:2]
y2 = new_df2.iloc[:,-1]
X_train2,X_test2,y_train,y_test = train_test_split(X2,y2,test_size=0.2,random_state=2)
lr2 = LinearRegression()
lr2.fit(X_train2,y_train)
y_pred2 = lr2.predict(X_test2)
new_r2 = r2_score(y_test,y_pred2)

ad2 = adjusted_r2(new_r2, X_test2)