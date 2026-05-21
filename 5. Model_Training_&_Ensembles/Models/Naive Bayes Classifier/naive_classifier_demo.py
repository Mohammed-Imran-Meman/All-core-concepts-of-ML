import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('play_tennis.csv')
df.drop(columns='day', inplace=True)
df.head()

counts=df['play'].value_counts(normalize=True) # avg
Yes_count = counts['Yes']
No_count = counts['No']


outlookplay = pd.crosstab(df['outlook'],df['play'],normalize='index') # frequency table of outlook and play
POvercastNo = outlookplay.at["Overcast","No"]
PRainNo = outlookplay.at["Rain","No"]
PSunnyNo = outlookplay.at["Sunny","No"]
POvercastYes = outlookplay.at["Overcast","Yes"]
PRainYes = outlookplay.at["Rain","Yes"]
PSunnyYes = outlookplay.at["Sunny","Yes"]

# frequency of all cols with play column
# List of all columns except the target 'play'
features = [col for col in df.columns if col != 'play']

for col in features:
  # Create the normalized crosstab
  ct = pd.crosstab(df[col], df['play'], normalize='index')
    
  # Loop through the rows (e.g., Sunny, Overcast) and columns (Yes, No)
  for row in ct.index:
    for col_val in ct.columns:
      # Create a clean variable name (e.g., "PSunnyYes")
      # We remove spaces/special chars just in case
      var_name = f"P{row}{col_val}".replace(" ", "")
            
      # Assign the value to that variable name in the global scope
      globals()[var_name] = ct.at[row, col_val]

# Finding probability of play using naive bayes classifier given Outlook=sunny, temp=hot, humidity=high, wind=weak

PYes = Yes_count*PSunnyYes*PHotYes*PHighYes*PWeakYes
PNo = No_count*PSunnyNo*PHotNo*PHighNo*PWeakNo