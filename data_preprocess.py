import pandas as pd
import numpy as np

d = pd.read_csv('data/train_new.csv')

# 日期
d['start_date'] = d['start_time'].str[:10]
d['end_date'] = d['end_time'].str[:10]

# to datetime
d['start_time'] = pd.to_datetime(d['start_time'])
d['end_time'] = pd.to_datetime(d['end_time'])

# hour
d['start_hour'] = d['start_time'].apply(lambda x: x.hour)
d['end_hour'] = d['end_time'].apply(lambda x: x.hour)

# is weekday or not
datetable = pd.date_range(start='2018-01-01', end='2018-09-01', freq='D')
datetable = pd.DataFrame(
    {
        'datetime': datetable,
        'date': datetable.astype(str)})
datetable['isweekday'] = datetable['datetime'].apply(
    lambda x: 1 if x.dayofweek < 5 else 0
)
# 各种假期及调班
add_holiday = (
    '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
    '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
    '2018-04-07', '2018-05-01', '2018-06-18')
add_weekday = (
    '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28'
)
datetable.loc[datetable['date'].isin(add_holiday), 'isweekday'] = 0
datetable.loc[datetable['date'].isin(add_weekday), 'isweekday'] = 1
# 距下一个工作日的天数
datetable['index'] = datetable.index
datetable.loc[datetable['isweekday'] == 0, 'index'] = np.nan
datetable['index'] = datetable['index'].fillna(method='bfill')
datetable['next'] = datetable['index'].shift(-1)
datetable['index'] = datetable.index
datetable['nextweekday'] = datetable['next'] - datetable['index']
datetable['nextweekday'].iloc[-2:] = [3, 2]  # 最后两天手动补上
datetable = datetable[['date', 'isweekday', 'nextweekday']]
# merge回去
datetable.columns = ['start_date', 'start_isweekday', 'start_nextweekday']
d = d.merge(datetable, 'left', on='start_date')
datetable.columns = ['end_date', 'end_isweekday', 'end_nextweekday']
d = d.merge(datetable, 'left', on='end_date')

d = d[
    [
        'r_key', 'start_hour', 'end_hour', 'start_isweekday',
        'end_isweekday', 'start_nextweekday', 'end_nextweekday'
    ]]
d.to_csv('train_time.csv', index=False, encoding='utf-8')

#################test#################################################
d = pd.read_csv('data/test_new.csv')

# 日期
d['start_date'] = d['start_time'].str[:10]

# to datetime
d['start_time'] = pd.to_datetime(d['start_time'])

# hour
d['start_hour'] = d['start_time'].apply(lambda x: x.hour)

# is weekday or not
datetable = pd.date_range(start='2018-01-01', end='2018-09-01', freq='D')
datetable = pd.DataFrame(
    {
        'datetime': datetable,
        'date': datetable.astype(str)})
datetable['isweekday'] = datetable['datetime'].apply(
    lambda x: 1 if x.dayofweek < 5 else 0
)
# 各种假期及调班
add_holiday = (
    '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
    '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
    '2018-04-07', '2018-05-01', '2018-06-18')
add_weekday = (
    '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28'
)
datetable.loc[datetable['date'].isin(add_holiday), 'isweekday'] = 0
datetable.loc[datetable['date'].isin(add_weekday), 'isweekday'] = 1
# 距下一个工作日的天数
datetable['index'] = datetable.index
datetable.loc[datetable['isweekday'] == 0, 'index'] = np.nan
datetable['index'] = datetable['index'].fillna(method='bfill')
datetable['next'] = datetable['index'].shift(-1)
datetable['index'] = datetable.index
datetable['nextweekday'] = datetable['next'] - datetable['index']
datetable['nextweekday'].iloc[-2:] = [3, 2]  # 最后两天手动补上
datetable = datetable[['date', 'isweekday', 'nextweekday']]
# merge回去
datetable.columns = ['start_date', 'start_isweekday', 'start_nextweekday']
d = d.merge(datetable, 'left', on='start_date')

d = d[['r_key', 'start_time', 'start_date', 'start_isweekday', 'start_nextweekday']]
d.to_csv('test_time.csv', index=False, encoding='utf-8')
