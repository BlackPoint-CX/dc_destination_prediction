import os
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import make_scorer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from common import EXTRACTED_DIR, getDistance, MODEL_DBSCAN_DIR, DATA_DIR, FILTER_DIR, getDistance_dbscan, \
    GENERATED_DIR, MODEL_RF_DIR
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

'''
train.csv r_key,out_id,start_time,end_time,start_lat,start_lon,end_lat,end_lon
test.csv r_key,out_id,start_time,start_lat,start_lon
There is no end_time in test.csv, so we should cluster by row_id to show the character of each row_id

'''

# original start time format : '2018-01-20 10:13:43'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')


def build_df(file_path, is_train):
    result_df = None
    if is_train:
        result_df = pd.read_csv(file_path, parse_dates=['start_time', 'end_time'], date_parser=dateparse,
                                dtype={'r_key': np.str, 'out_id': np.str, 'start_time': np.str, 'end_time': np.str,
                                       'start_lat': np.float, 'start_lon': np.float, 'end_lat': np.float,
                                       'end_lon': np.float})
    else:
        result_df = pd.read_csv(file_path, parse_dates=['start_time'], date_parser=dateparse,
                                dtype={'r_key': np.str, 'out_id': np.str, 'start_time': np.str, 'end_time': np.str,
                                       'start_lat': np.float, 'start_lon': np.float})
    return result_df


def build_df_generated(file_path, is_train):
    if is_train:
        result_df = pd.read_csv(file_path, parse_dates=['start_time', 'end_time'], date_parser=dateparse,
                                dtype={'r_key': np.str, 'out_id': np.str, 'start_time': np.str, 'end_time': np.str,
                                       'start_lat': np.float, 'start_lon': np.float, 'end_lat': np.float,
                                       'end_lon': np.float, 'start_label': np.int32, 'end_label': np.int32})
    else:
        result_df = pd.read_csv(file_path, parse_dates=['start_time'], date_parser=dateparse,
                                dtype={'r_key': np.str, 'out_id': np.str, 'start_time': np.str,
                                       'start_lat': np.float, 'start_lon': np.float, 'start_label': np.int32})
    return result_df


def is_working_day(s_weekday):
    """
    判断是否是工作日, 如果是工作日, 返回1, 否则返回0.
    :param s_weekday:
    :return:
    """
    if s_weekday in [5, 6]:
        return 1
    else:
        return 0


def build_feature(dataframe, is_train):
    dataframe['s_hour'] = dataframe['start_time'].map(lambda start_time: start_time.hour)
    dataframe['s_year'] = dataframe['start_time'].map(lambda start_time: start_time.year)
    dataframe['s_month'] = dataframe['start_time'].map(lambda start_time: start_time.month)
    dataframe['s_day'] = dataframe['start_time'].map(lambda start_time: start_time.day)
    dataframe['s_weekday'] = dataframe['start_time'].map(lambda start_time: start_time.weekday())
    dataframe['s_is_working_day'] = dataframe['s_weekday'].map(lambda s_weekday: is_working_day(s_weekday))

    if is_train:
        # 计算训练集中每次行程距离
        # dataframe['distance'] = dataframe.apply(
        #     lambda row: getDistance(row['start_lat'], row['start_lon'], row['end_lat'], row['end_lon']),axis=1)
        # 计算训练集中每次行程时间
        dataframe['period'] = dataframe[['start_time', 'end_time']].apply(
            func=lambda row: (row['end_time'] - row['start_time']).seconds // 60, axis=1)

        # 用车频率, 行程长度, 出行时间, 起止点数量, 行程角度, 跨省市行程占比, 白天里程占比, 工作日里程占比, 行程方差, 行程长度平均数, 上一次的行程起止点, 时间间隔, 距离
        # dataframe['trip_freq'] = None
        #
        # dataframe['last_trip_start_lat'] = None
        # dataframe['last_trip_start_lon'] = None
        # dataframe['last_trip_period'] = None
        # dataframe['last_trip_distance'] = None

    return dataframe


def clean_unnecessary_car_id(ori_train_df, ori_test_df):
    train_file_path = os.path.join(EXTRACTED_DIR, 'train.csv')
    test_file_path = os.path.join(EXTRACTED_DIR, 'test.csv')
    ori_train_df = build_df(train_file_path, True)
    ori_test_df = build_df(test_file_path, False)

    valiable_out_id_train = set(ori_train_df['out_id'])  # 5817
    valiable_out_id_test = set(ori_test_df['out_id'])  # 5033

    def check_out_id_valuable(row):
        if row['out_id'] in valiable_out_id_test:
            return True
        else:
            return False

    ori_train_df['in_test'] = ori_train_df.apply(lambda row: check_out_id_valuable(row), axis=1)

    valid_train_df = ori_train_df[ori_train_df['in_test'] == True]
    valid_train_df.drop(['in_test'], axis=1).to_csv(os.path.join(FILTER_DIR, 'filter_train.csv'), index=False,
                                                    mode='w+')
    ori_test_df.to_csv(os.path.join(FILTER_DIR, 'filter_test.csv'), index=False,
                       mode='w+')


def build_df_filter():
    """
    :return:
    """
    train_file_path = os.path.join(FILTER_DIR, 'filter_train.csv')
    test_file_path = os.path.join(FILTER_DIR, 'filter_test.csv')
    train_df = build_df(train_file_path, True)
    test_df = build_df(test_file_path, False)

    valid_out_id = set(test_df['out_id'])
    valid_out_id_len = len(valid_out_id)

    train_out_id_start_pos = train_df[['out_id', 'start_lat', 'start_lon']]
    train_out_id_start_pos.columns = ['out_id', 'pos_lat', 'pos_lon']  # (1432069, 3)

    train_out_id_end_pos = train_df[['out_id', 'end_lat', 'end_lon']]
    train_out_id_end_pos.columns = ['out_id', 'pos_lat', 'pos_lon']  # (1432069, 3)

    test_out_id_start_pos = test_df[['out_id', 'start_lat', 'start_lon']]
    test_out_id_start_pos.columns = ['out_id', 'pos_lat', 'pos_lon']  # (47493, 3)

    out_id_pos_union_all = pd.concat(
        [train_out_id_start_pos, train_out_id_end_pos, test_out_id_start_pos])  # (2911631, 3)

    new_train_df = pd.DataFrame()
    new_test_df = pd.DataFrame()
    out_id_pos_label_df = pd.DataFrame(columns=['label', 'out_id', 'pos_lat', 'pos_lon'])

    for index, out_id in enumerate(valid_out_id):
        # for out_id in ['868260020906435', '891631605004051']:
        print('now processing {}/{} outlook_id : {}'.format(index, valid_out_id_len, out_id))
        specific_out_id_pos = out_id_pos_union_all[out_id_pos_union_all['out_id'] == out_id]

        db_model = DBSCAN(min_samples=1, eps=100, metric=getDistance_dbscan, n_jobs=-1)
        specific_out_id_pos['pos_label'] = db_model.fit_predict(specific_out_id_pos[['pos_lat', 'pos_lon']])

        # 将 train_df 中的start_lat, start_lon, end_lat, end_lon 转换为聚类得出的label
        out_id_train_df = train_df[train_df['out_id'] == out_id]
        out_id_train_df = out_id_train_df.merge(specific_out_id_pos, how='left',
                                                left_on=['out_id', 'start_lat', 'start_lon'],
                                                right_on=['out_id', 'pos_lat', 'pos_lon'])
        out_id_train_df['start_label'] = out_id_train_df['pos_label']
        out_id_train_df.drop(['pos_lat', 'pos_lon', 'pos_label'], inplace=True, axis=1)

        out_id_train_df = out_id_train_df.merge(specific_out_id_pos, how='left',
                                                left_on=['out_id', 'end_lat', 'end_lon'],
                                                right_on=['out_id', 'pos_lat', 'pos_lon'])
        out_id_train_df['end_label'] = out_id_train_df['pos_label']
        out_id_train_df.drop(['pos_lat', 'pos_lon', 'pos_label'], inplace=True, axis=1)

        # 将 test_df 中的start_lat, start_lon, end_lat, end_lon 转换为聚类得出的label
        out_id_test_df = test_df[test_df['out_id'] == out_id]
        out_id_test_df = out_id_test_df.merge(specific_out_id_pos, how='left',
                                              left_on=['out_id', 'start_lat', 'start_lon'],
                                              right_on=['out_id', 'pos_lat', 'pos_lon'])
        out_id_test_df['start_label'] = out_id_test_df['pos_label']
        out_id_test_df.drop(['pos_lat', 'pos_lon', 'pos_label'], inplace=True, axis=1)

        # 汇总 labelled 过的新 train 和 test 数据集
        new_train_df = new_train_df.append(out_id_train_df)
        new_test_df = new_test_df.append(out_id_test_df)

        # 将 DBSCAN 标记的位置环境和label进行记录
        label_pos = specific_out_id_pos.groupby('pos_label').mean()
        label_pos['out_id'] = out_id
        label_pos['label'] = label_pos.index
        print('{} {}'.format(out_id, label_pos.shape))
        out_id_pos_label_df = pd.concat([out_id_pos_label_df, label_pos])

    out_id_pos_label_df.to_csv(os.path.join(GENERATED_DIR, 'out_id_post_label.csv'), index=False, mode='w+')
    new_train_df.to_csv(os.path.join(GENERATED_DIR, 'new_train_df.csv'), index=False, mode='w+')
    new_test_df.to_csv(os.path.join(GENERATED_DIR, 'new_test_df.csv'), index=False, mode='w+')


def predict():
    test_file_path = os.path.join(GENERATED_DIR, 'new_test_df.csv')
    test_df = build_df_generated(test_file_path, False)

    out_id_post_label_df = pd.read_csv(os.path.join(GENERATED_DIR, 'out_id_post_label.csv'),
                                       dtype={'label': np.int32, 'out_id': np.str, 'pos_lat': np.float,
                                              'pos_lon': np.float})
    features = ['s_month', 's_day', 's_weekday', 's_is_working_day', 'start_label']

    result_features = ['r_key', 'pos_lat', 'pos_lon']
    result_df = pd.DataFrame(columns=result_features)

    valid_out_id = set(test_df['out_id'])
    valid_out_id_len = len(valid_out_id)
    for index, out_id in enumerate(valid_out_id):
        print('Processing {}/{}'.format(index, valid_out_id_len))
        model = joblib.load(os.path.join(MODEL_RF_DIR, '{}.mdl'.format(out_id)))
        # make prediction on test dataset
        part_test_df = test_df[test_df['out_id'] == out_id]
        part_test_df = build_feature(part_test_df, False)
        part_test_df['predict_label'] = model.predict(part_test_df[features].values)
        part_test_df = pd.merge(left=part_test_df, right=out_id_post_label_df, left_on=['out_id', 'predict_label'],
                                right_on=['out_id', 'label'], how='left')
        result_df = pd.concat([result_df, part_test_df[result_features]])

    result_df.columns = ['r_key', 'end_lat', 'end_lon']
    result_df.to_csv(os.path.join(DATA_DIR, 'result_2018-11-19 12:00:00.csv'), index=False)


if __name__ == '__main__':
    # build_df_filter()
    build_model()
    predict()
