import numpy as np
import pandas as pd
import datetime

date = pd.read_csv('orders.csv')[['date']]
date.head()
time = pd.read_csv('messages.csv')[['date']]
time.head()

date['date'] = pd.to_datetime(date['date'])
date.info()

date['date_day'] = date['date'].dt.day
date.sample(5)
date['date_month_number'] = date['date'].dt.month
date.sample(5)
date['date_year'] = date['date'].dt.year
date.sample(5)
date['date_day_name'] = date['date'].dt.day_name()
date.sample(5)
date['date_month_name'] = date['date'].dt.month_name()
date.sample(5)
date['date_dayOfWeek'] = date['date'].dt.dayofweek + 1
date.sample(5)
date['date_quarter'] = date['date'].dt.quarter
date.sample(5)
date['date_weekend'] = np.where(date['date'].dt.dayofweek.isin([5,6]),1,0)
date.sample(10)
date['date_quarter'] = date['date'].dt.quarter
date.sample(5)

today = datetime.datetime.today()
today - date['date']
(today - date['date']).dt.days
np.round((today -date['date']) / np.timedelta64(1, 'W'),0)
np.round((today -date['date']) / np.timedelta64(1, 'D'),0)
np.round((today -date['date']) / np.timedelta64(1, 'h'),0)
np.round((today -date['date']) / np.timedelta64(1, 'm'),0)
np.round((today -date['date']) / np.timedelta64(1, 's'),0)
np.round((today -date['date']) / np.timedelta64(1, 'ms'),0)

time['date'] = pd.to_datetime(time['date'])
time.info()
time['time'] = time['date'].dt.time
time.head()
time['minute'] = time['date'].dt.minute
time.head()
time['hour'] = time['date'].dt.hour
time.head()
time['second'] = time['date'].dt.second
time.head()