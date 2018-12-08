# 简单的方案，就是先统计用户星期最喜欢去的地方，之后对这些地方标记一下，如果未来真的去过，标记1否则0
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')
from math import radians, atan, tan, sin, acos, cos


def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001


def f(d):
    """
    计算经纬度的球面距离
    :param d:
    :return:
    """
    return 1 / (1 + np.exp(-(d - 1000) / 250))


# 计算误差值
def getDistanceFromDF(data):
    tmp = data[['end_lat', 'end_lon', 'predict_end_lat', 'predict_end_lon']].astype(float)
    error = []
    for i in tmp.values:
        t = getDistance(i[0], i[1], i[2], i[3])
        error.append(t)

    print(np.sum(f(np.array(error))) / tmp.shape[0])


def dateConvert(data, isTrain):
    print('convert string to datetime')
    data['start_time'] = pd.to_datetime(data['start_time'])
    if isTrain:
        data['end_time'] = pd.to_datetime(data['end_time'])
    data['weekday'] = data['start_time'].dt.weekday + 1
    return data


#
def latitude_longitude_to_go(data, isTrain):
    tmp = data[['start_lat', 'start_lon']]
    start_geohash = []
    for t in tmp.values:
        start_geohash.append(str(round(t[0], 5)) + '_' + str(round(t[1], 5)))
    data['startGo'] = start_geohash
    if isTrain:
        tmp = data[['end_lat', 'end_lon']]
        end_geohash = []
        for t in tmp.values:
            end_geohash.append(str(round(t[0], 5)) + '_' + str(round(t[1], 5)))
        data['endGo'] = end_geohash
    return data


# 用户去过最多的三个地方
def getMostTimesCandidate(candidate):
    mostTimeCandidate = candidate[candidate['start_time'] <= '2018-06-30 23:59:59']
    mostTimeCandidate = mostTimeCandidate[['out_id', 'endGo', 'end_lat', 'end_lon', 'weekday']]
    mostTimeCandidate_3 = mostTimeCandidate.groupby(['out_id', 'endGo', 'weekday'], as_index=False)['endGo'].agg(
        {'mostCandidateCount': 'count'})
    mostTimeCandidate_3.sort_values(['mostCandidateCount', 'out_id'], inplace=True, ascending=False)
    mostTimeCandidate_3 = mostTimeCandidate_3.groupby(['out_id', 'weekday']).tail(7)
    return mostTimeCandidate_3


# 将预测出的位置字符串分割为经纬度
def geoHashToLatLoc(data):
    tmp = data[['endGo']]
    predict_end_lat = []
    predict_end_lon = []
    for i in tmp.values:
        lats, lons = str(i[0]).split('_')
    predict_end_lat.append(lats)
    predict_end_lon.append(lons)
    data['predict_end_lat'] = predict_end_lat
    data['predict_end_lon'] = predict_end_lon
    return data


def calcGeoHasBetween(go1, go2):
    latA, lonA = str(go1).split('_')
    latB, lonB = str(go2).split('_')
    distence = getDistance(float(latA), float(lonA), float(latB), float(lonB))
    return distence


# start to end distance
def calcGeoHasBetweenMain(data):
    distance = []
    tmp = data[['endGo', 'startGo']]
    for i in tmp.values:
        distance.append(calcGeoHasBetween(i[0], i[1]) / 1000)
    data['distance'] = distance
    return data


def analysis():
    train = pd.read_csv('../data/extracted/train.csv')
    test = pd.read_csv('../data/extracted/test.csv')
    print(train['start_time'].min(), train['start_time'].max())  # 2018-01-01 00:01:58 2018-08-01 00:49:30
    print(train[train['start_time'] > '2018-06-30 23:59:59'].shape)  # (157618, 8)
    print(train[train['start_time'] <= '2018-06-30 23:59:59'].shape)  # (1290703, 8)

    print(test['start_time'].min(), test['start_time'].max())  # 2018-07-01 00:01:19 2018-07-31 23:59:22
    print(test.shape)  # (47493, 5)
    trainIndex = train.shape[0]
    testIndex = test.shape[0]
    print(trainIndex, testIndex)  # 1448321 47493


def train():
    print('begin')
    # 用1-6月去提取最常去的地方, 用 7 月去训练
    train = pd.read_csv('../data/extracted/train.csv')
    test = pd.read_csv('../data/extracted/test.csv')

    train = dateConvert(train, True)
    test = dateConvert(test, False)
    train = latitude_longitude_to_go(train, True)
    test = latitude_longitude_to_go(test, False)
    train.to_csv('train1.csv', index=False)
    test.to_csv('test1.csv', index=False)

    userMostTimes3loc = getMostTimesCandidate(train)
    val = train[train['start_time'] > '2018-06-30 23:59:59'] # 用7月数据进行validation
    val = val[['r_key', 'out_id', 'end_lat', 'end_lon', 'weekday', 'startGo', 'endGo', 'start_lat', 'start_lon']]
    val.rename(columns={'endGo': 'trueEndGo'}, inplace=True)
    val = pd.merge(val, userMostTimes3loc, on=['out_id', 'weekday'], how='left', copy=False)
    val['endGo'] = val['endGo'].fillna(val['startGo'])
    val['flag1'] = val['trueEndGo'] == val['endGo']
    val['flag1'] = val['flag1'].astype(int)
    val = calcGeoHasBetweenMain(val)
    test = test[['r_key', 'out_id', 'weekday', 'startGo', 'start_lat', 'start_lon']]
    test = pd.merge(test, userMostTimes3loc, on=['out_id', 'weekday'], how='left', copy=False)
    test['endGo'] = test['endGo'].fillna(test['startGo'])
    test = calcGeoHasBetweenMain(test)
    # model
    feature = ['start_lat', 'start_lon', 'weekday', 'distance', 'mostCandidateCount']

    print('training')
    lr = LogisticRegression()
    lr.fit(val[feature].fillna(-1).values, val['flag1'].values)
    pre = lr.predict_proba(val[feature].fillna(-1).values)[:, 1]
    val_result = val[['r_key', 'endGo', 'end_lat', 'end_lon', ]]
    val_result['predict'] = pre
    val_result = val_result.sort_values(['predict'], ascending=False)
    val_result = val_result.drop_duplicates(['r_key'])
    val = geoHashToLatLoc(val)
    getDistanceFromDF(val)
    subPre = lr.predict_proba(test[feature].fillna(-1).values)[:, 1]
    test_result = test[['r_key', 'endGo']]
    test_result['predict'] = subPre
    test_result = test_result.sort_values(['predict'], ascending=False)
    test_result = test_result.drop_duplicates(['r_key'])
    test_result = geoHashToLatLoc(test_result)
    submit = test_result[['r_key', 'predict_end_lat', 'predict_end_lon']]
    submit.columns = ['r_key', 'end_lat', 'end_lon']
    submit.to_csv('./result_2018-11-19 12:00:00.csv', index=False)
