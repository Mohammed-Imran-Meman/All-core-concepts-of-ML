import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('placement.csv')
df.head()
df.describe()

plt.figure(figsize=(14,5))
plt.subplot(121)
sns.distplot(df['cgpa'])
plt.title("CGPA")

plt.subplot(122)
sns.distplot(df['placement_exam_marks'])
plt.title("Placement Exam Marks")

upper_value = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_value = df['cgpa'].mean() - 3*df['cgpa'].std()

print("Highest Allowed",upper_value)
print("Lowest Allowed",lower_value)

df_outliers = df[(df['cgpa']>upper_value) | (df['cgpa']<lower_value)]

# trimming
df_trimmed = df[(df['cgpa']<=upper_value) & (df['cgpa']>=lower_value)]
df_trimmed.describe()

# using z-score formula
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean())/df['cgpa'].std()
df_zscore_trimmed = df[(df['cgpa_zscore']<=3) & (df['cgpa_zscore']>=-3)]
df_zscore_trimmed.describe()

# Capping
df['cgpa'] = np.where(
  df['cgpa']>upper_value, 
  upper_value,
  np.where(
    df['cgpa']<lower_value,
    lower_value,
    df['cgpa']
  )
)
df.describe()