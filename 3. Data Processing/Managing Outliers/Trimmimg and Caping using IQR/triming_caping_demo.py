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

df['placement_exam_marks'].plot(kind='box')

q1 = df['placement_exam_marks'].quantile(0.25)
q3 = df['placement_exam_marks'].quantile(0.75)
iqr = q3 - q1
upper_value = q3 + 1.5*iqr
lower_value = q1 - 1.5*iqr

print("Highest Allowed",upper_value)
print("Lowest Allowed",lower_value)

df_outliers = df[(df['placement_exam_marks']>upper_value) | (df['placement_exam_marks']<lower_value)]

# trimming
df_trimmed = df[(df['placement_exam_marks']<=upper_value) & (df['placement_exam_marks']>=lower_value)]
df_trimmed.describe()
df_trimmed['placement_exam_marks'].plot(kind='box')

# Capping
df['placement_exam_marks'] = np.where(
  df['placement_exam_marks']>=upper_value, 
  upper_value,
  np.where(
    df['placement_exam_marks']<=lower_value,
    lower_value,
    df['placement_exam_marks']
  )
)
df.describe()
df['placement_exam_marks'].plot(kind='box')