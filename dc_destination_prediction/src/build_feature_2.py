import os
from collections import defaultdict
from logging import warning
from multiprocessing.pool import Pool

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import numpy as np
from build_feature import build_df_generated, build_feature
from common import GENERATED_DIR, getDistance, MODEL_RF_DIR, DATA_DIR

features = ['s_hour', 's_month', 's_day', 's_weekday', 's_is_working_day', 'start_label']


def get_sample_out_id(train_df, sample_size=100):
    """
    从train_df中获取每辆车的历史记录, 进行分层抽样, 返回sample_out_id
    :param train_df:
    :param sample_size: 抽取样本数
    :return:
    """
    inner_df = train_df.groupby(['out_id'], as_index=False)['out_id'].agg({'recordsNo': 'count'})
    X_train, X_test = train_test_split(inner_df, train_size=sample_size)
    return X_train['out_id'].tolist()


def get_sample_df(train_df, sample_size=100):
    """
    从train_df中抽样部分out_id,并根据out_id抽取train_df的对应记录,组成抽样记录进行返回.
    :param train_df:
    :param test_df:
    :param sample_size:
    :return:
    """
    sample_out_id = get_sample_out_id(train_df, sample_size)
    print('sample_out_id.length {}'.format(len(sample_out_id)))
    sample_train_df = train_df[train_df['out_id'].isin(sample_out_id)]
    print('sample_train_df.shape {}'.format(sample_train_df.shape))
    return sample_train_df


def build_single_out_id(out_id, sample_train_df, sample_valid_df, out_id_post_label_df):
    try:
        # 构建模型
        single_train_df = sample_train_df[sample_train_df['out_id'] == out_id]
        # single_valid_df = sample_valid_df[sample_valid_df['out_id'] == out_id]

        single_train_df_X = single_train_df[features].values
        single_train_df_y = single_train_df['end_label'].values

        model_params_list = [
            {'model_name': 'RF', 'model': RandomForestClassifier(),
             'parameters': {'n_estimators': list(range(75, 90)), 'max_depth': [1, 2, 3, 4]}},
        ]

        model_list = []

        def custom_score(y_true, y_pred):
            temp_df = pd.DataFrame(columns=['y_true', 'y_pred'])
            temp_df['y_true'] = y_true
            temp_df['y_pred'] = y_pred
            temp_df['out_id'] = out_id

            part_valid_df = pd.merge(left=temp_df, right=out_id_post_label_df, left_on=['out_id', 'y_true'],
                                     right_on=['out_id', 'label'], how='left')
            part_valid_df.rename(columns={'pos_lat': 'end_lat', 'pos_lon': 'end_lon', 'label': 'end_label'},
                                 inplace=True)

            part_valid_df = pd.merge(left=part_valid_df, right=out_id_post_label_df,
                                     left_on=['out_id', 'y_pred'],
                                     right_on=['out_id', 'label'], how='left')
            part_valid_df.rename(columns={'label': 'pos_label'}, inplace=True)

            part_valid_df['err_dis'] = part_valid_df.apply(
                lambda row: getDistance(row['end_lat'], row['end_lon'], row['pos_lat'], row['pos_lon']), axis=1)
            error = np.mean(part_valid_df['err_dis'])
            return error

        my_scoring = make_scorer(custom_score, greater_is_better=False)

        for model_params_dict in model_params_list:
            model_name = model_params_dict['model_name']
            model = model_params_dict['model']
            parameters = model_params_dict['parameters']
            print('{} {}'.format(model_name, parameters))
            clf = GridSearchCV(estimator=model, param_grid=parameters, scoring=my_scoring, cv=3)
            clf.fit(single_train_df_X, single_train_df_y)

            best_estimator = clf.best_estimator_
            best_params = clf.best_params_
            best_score = clf.best_score_

            model_list.append([out_id, model_name, best_estimator, best_params, best_score])
            joblib.dump(value=best_estimator,
                        filename=os.path.join(MODEL_RF_DIR, '{}_{}_{}.mdl'.format(out_id, model_name, int(best_score))))

        with open(os.path.join(DATA_DIR, 'model_result.txt'), 'a') as w_file:
            for line in model_list:
                out_id, model_name, best_estimator, best_params, best_score = line
                w_file.write(str([out_id, best_params, best_score]) + '\n')
    except Exception:
        print('wrong {}'.format(out_id))


def build_model():
    train_file_path = os.path.join(GENERATED_DIR, 'new_train_df.csv')
    train_df = build_df_generated(train_file_path, True)
    train_df = build_feature(train_df, True)

    # sample_df = get_sample_df(train_df, 100)
    sample_df = train_df

    sample_train_df = sample_df[sample_df['start_time'] < '2018-06-30 23:59:59']
    sample_valid_df = sample_df[sample_df['start_time'] > '2018-06-30 23:59:59']

    print('sample_train_df.shape {}'.format(sample_train_df.shape))
    print('sample_valid_df.shape {}'.format(sample_valid_df.shape))

    out_id_post_label_df = pd.read_csv(os.path.join(GENERATED_DIR, 'out_id_post_label.csv'),
                                       dtype={'label': np.int32, 'out_id': np.str, 'pos_lat': np.float,
                                              'pos_lon': np.float})
    # out_id_post_label_df['label'] = out_id_post_label_df.index
    # out_id_post_label_df.reset_index()

    sample_out_id_list = sample_df['out_id'].drop_duplicates().tolist()
    print('Before rmv ,sample_out_id_list len : {}'.format(len(sample_out_id_list)))
    already_list = get_already_trained_out_id()
    for ele in already_list:
        if ele in sample_out_id_list:
            sample_out_id_list.remove(ele)
    sample_out_id_list_len = len(sample_out_id_list)
    print('After rmv ,sample_out_id_list len : {}'.format(len(sample_out_id_list)))
    # out_id = sample_out_id_list[24]  # 单挑一个out_id进行后续试验.
    pool = Pool(10)

    for idx, out_id in enumerate(sample_out_id_list):
        print('Processing {}/{}'.format(idx, sample_out_id_list_len))
        # build_single_out_id(out_id, sample_train_df, sample_valid_df, out_id_post_label_df)
        pool.apply_async(func=build_single_out_id,
                         args=(out_id, sample_train_df, sample_valid_df, out_id_post_label_df))

    pool.close()
    pool.join()

    print('Finished.')


def inference():
    # 载入test_df
    test_df = build_df_generated(os.path.join(GENERATED_DIR, 'new_test_df.csv'), False)
    test_df = build_feature(test_df, False)
    # 载入out_id, position_label对应关系
    out_id_post_label_df = pd.read_csv(os.path.join(GENERATED_DIR, 'out_id_post_label.csv'),
                                       dtype={'label': np.int32, 'out_id': np.str, 'pos_lat': np.float,
                                              'pos_lon': np.float})

    model_dict = {}
    for file_name in os.listdir(MODEL_RF_DIR):
        out_id, model_name, error = file_name.strip()[:-4].split('_')
        model_dict[out_id] = os.path.join(MODEL_RF_DIR, file_name)

    out_id_list = set(test_df['out_id'].values.tolist())
    out_id_list_len = len(out_id_list)
    print('out_id_list_len : %d ' % out_id_list_len)

    result_df = pd.DataFrame()
    for index, out_id in enumerate(out_id_list):
        print('Processing {}/{} Out_id : {}'.format(index, out_id_list_len, out_id))
        try:
            model = joblib.load(model_dict[out_id])
            part_df = test_df[test_df['out_id'] == out_id]

            # make prediction on valid dataset and evaluate
            if not part_df.empty:
                part_df['predict_label'] = model.predict(part_df[features].values)

                part_df = pd.merge(left=part_df, right=out_id_post_label_df,
                                   left_on=['out_id', 'predict_label'],
                                   right_on=['out_id', 'label'], how='left')

                # part_df['err_dis'] = part_df.apply(
                #     lambda row: getDistance(row['end_lat'], row['end_lon'], row['pos_lat'], row['pos_lon']), axis=1)

                result_df = pd.concat([result_df, part_df])
        except KeyError as e:
            part_df = test_df[test_df['out_id'] == out_id]

            # make prediction on valid dataset and evaluate
            if not part_df.empty:
                part_df = pd.merge(left=part_df, right=out_id_post_label_df,
                                   left_on=['out_id', 'predict_label'],
                                   right_on=['out_id', 'label'], how='left')

                # part_df['err_dis'] = part_df.apply(
                #     lambda row: getDistance(row['end_lat'], row['end_lon'], row['pos_lat'], row['pos_lon']), axis=1)

                result_df = pd.concat([result_df, part_df])

    result_df.to_csv(os.path.join(DATA_DIR, 'result_df.csv'))


def get_already_trained_out_id():
    already_list = []
    mdl_list = os.listdir(MODEL_RF_DIR)
    for file_name in mdl_list:
        file_main = file_name.split('.')[0]
        out_id, model_type, error = file_main.split('_')
        already_list.append(out_id)

    return already_list


def test():
    out_id_dict = defaultdict(dict)
    remove_list = []

    file_list = os.listdir(MODEL_RF_DIR)
    print(len(file_list))

    for line in file_list:
        out_id, model_type, distance_error = line.split('.')[0].split('_')
        if out_id in out_id_dict:
            if distance_error > out_id_dict[out_id]['distance_error']:
                remove_list.append(line)
            else:
                old_file_name = '{}_RF_{}.mdl'.format(out_id_dict[out_id]['out_id'],
                                                      out_id_dict[out_id]['distance_error'])
                remove_list.append(old_file_name)
                out_id_dict[out_id] = {'out_id': out_id, 'params_dict': model_type, 'distance_error': distance_error}
        else:
            out_id_dict[out_id] = {'out_id': out_id, 'params_dict': model_type, 'distance_error': distance_error}

    print(len(out_id_dict))
    print(len(remove_list))

    for file_name in remove_list:
        try:
            os.remove(os.path.join(MODEL_RF_DIR, file_name))
        except FileNotFoundError:
            print('Not found %s' % file_name)


def diff(test_file_path, result_file_path):
    test_df = pd.read_csv(test_file_path)
    result_df = pd.read_csv(result_file_path)


    test_out_id_set = set(test_df['out_id'].values.tolist())
    result_out_id_set = set(result_df['out_id'].values.tolist())

    print(len(test_out_id_set))
    print(len(result_out_id_set))

    for ele in test_out_id_set:
        if ele not in result_out_id_set:
            print(ele)


def merge(test_file_path,result_file_path):
    test_df = pd.read_csv(test_file_path)
    result_df = pd.read_csv(result_file_path)


    test_df = test_df[['r_key','out_id','start_time','start_lat','start_lon']]
    print('test_df.shape : {}'.format(test_df.shape))

    result_df = result_df[['r_key','pos_lat','pos_lon']]
    result_df.columns = ['r_key','end_lat','end_lon']
    print('result_df.shape : {}'.format(result_df.shape))

    result_all_df = pd.merge(left=test_df,right=result_df,how='left',on=['r_key'])
    print('result_all_df.shape : {}'.format(result_all_df.shape))


    print('result_all_df end_lat null count : {}'.format(result_all_df['end_lat'].isnull().sum()))
    print('result_all_df end_lon null count : {}'.format(result_all_df['end_lon'].isnull().sum()))
    result_all_df['end_lat'].fillna(result_all_df['start_lat'],inplace=True)
    result_all_df['end_lon'].fillna(result_all_df['start_lon'],inplace=True)
    print('result_all_df end_lat null count : {}'.format(result_all_df['end_lat'].isnull().sum()))
    print('result_all_df end_lon null count : {}'.format(result_all_df['end_lon'].isnull().sum()))

    result_all_df.to_csv(os.path.join(DATA_DIR,'result_all_df.csv'),index=False)

if __name__ == '__main__':
    # build_model()
    # inference()
    # get_already_trained_out_id()
    # test()

    test_file_path = os.path.join(GENERATED_DIR,'new_test_df.csv')
    result_file_path = os.path.join(DATA_DIR,'result_df.csv')
    # diff(test_file_path,result_file_path)

    merge(test_file_path,result_file_path)
    #
    final_result_df= pd.read_csv(os.path.join(DATA_DIR,'result_all_df.csv'))
    final_result_df[['r_key','end_lat','end_lon']].to_csv(os.path.join(DATA_DIR,'final_result.csv'),index=False)

