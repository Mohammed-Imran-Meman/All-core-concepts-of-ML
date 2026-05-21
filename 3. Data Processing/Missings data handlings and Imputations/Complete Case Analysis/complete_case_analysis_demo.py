import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_science_job.csv')
df.head()
df.shape
df.isnull().mean()*100
cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]

df[cols].sample(5)

len(df[cols].dropna()) / len(df)

new_df = df[cols].dropna()
new_df.shape

new_df.hist(bins=50, density=True, figsize=(12, 12))
plt.show()

fig, (ax,ay) = plt.subplots(1, 2, figsize=(10, 4))

df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')
new_df['training_hours'].hist(bins=50, ax=ax, density=True, color='green')
df['training_hours'].plot.density(ax=ay, color='red')
new_df['training_hours'].plot.density(ax=ay, color='green')


fig, (ax,ay) = plt.subplots(1, 2, figsize=(10, 4))

df['experience'].hist(bins=50, ax=ax, density=True, color='red')
new_df['experience'].hist(bins=50, ax=ax, density=True, color='green')
df['experience'].plot.density(ax=ay, color='red')
new_df['experience'].plot.density(ax=ay, color='green')


fig, (ax,ay) = plt.subplots(1, 2, figsize=(10, 4))

df['city_development_index'].hist(bins=50, ax=ax, density=True, color='red')
new_df['city_development_index'].hist(bins=50, ax=ax, density=True, color='green')
df['city_development_index'].plot.density(ax=ay, color='red')
new_df['city_development_index'].plot.density(ax=ay, color='green')




