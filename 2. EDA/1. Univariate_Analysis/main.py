import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
df.head(5)
sns.countplot(df['Pclass'])
df['Pclass'].value_counts().plot(kind='bar')
df['Pclass'].value_counts().plot(kind='pie',autopct='%.2f')
plt.hist(df['Age'], bins=50)
sns.distplot(df['Age'])
sns.boxplot(df["Fare"])
df['Age'].min()
df['Age'].max()
df['Age'].mean()