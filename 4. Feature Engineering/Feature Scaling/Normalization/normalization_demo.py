import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('wine_data.csv', header=None,usecols=[0,1,2])
df.columns = ['Class label', 'Alcohol', 'Malic Acid']
sns.kdeplot(df['Alcohol'])

colour = {
  1: 'red',
  2: 'blue',
  3: 'purple'
}

sns.scatterplot(
  data=df,
  x='Alcohol',
  y='Malic Acid', 
  hue='Class label', 
  palette = colour
)

X_train,X_test,y_train,y_test = train_test_split(df.drop('Class label', axis=1), df['Class label'], test_size=0.3, random_state=0)

X_train.shape,X_test.shape

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_train.columns)

np.round(X_train.describe(),1)
np.round(X_train_scaled.describe(),1)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5)) 
ax1.scatter(X_train['Alcohol'],X_train['Malic Acid'], c=y_train)
ax1.set_title('Before Scaling')
ax2.scatter(X_train_scaled['Alcohol'],X_train_scaled['Malic Acid'],c=y_train)
ax2.set_title('After Scaling')

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5))
sns.kdeplot(X_train['Alcohol'],ax=ax1)
sns.kdeplot(X_train['Malic Acid'],ax=ax1)
ax1.set_title('Before Scaling')
sns.kdeplot(X_train_scaled['Alcohol'],ax=ax2)
sns.kdeplot(X_train_scaled['Malic Acid'],ax=ax2)
ax1.set_title('After Scaling')
