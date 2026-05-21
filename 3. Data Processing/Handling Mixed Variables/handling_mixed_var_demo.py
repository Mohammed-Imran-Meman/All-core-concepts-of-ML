import numpy as np
import pandas as pd

df = pd.read_csv('titanic.csv')
df.head()

df['number_numerical'] = pd.to_numeric(df["number"],errors='coerce',downcast='integer')
df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan)

df['cabin_numerical'] = df['Cabin'].str.extract('(\d+)')
df['cabin_categorical'] = df['Cabin'].str[0]

df['Ticket_numerical'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['Ticket_numerical'] = pd.to_numeric(df['Ticket_numerical'],errors='coerce',downcast='integer')

df['Ticket_categorical'] = df['Ticket'].apply(lambda s: s.split()[0])
df['Ticket_categorical'] = np.where(df['Ticket_categorical'].str.isdigit(), np.nan, df['Ticket_categorical'])