# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:45:50 2018

@author: MSIK
"""

import pandas as pd
import numpy as np
from datetime import date

import warnings
warnings.filterwarnings('ignore')

import lightgbm
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_predict

# 用户短信记录数据
smsTrain = pd.read_table('../data/sms_train.txt', header = None)
smsTrain.columns = ['uid','sms_opp_num','sms_opp_head','sms_opp_len','sms_start_time',
                    'sms_in_out']

smsTest = pd.read_table('../data/sms_test_a.txt', header = None)
smsTest.columns = ['uid','sms_opp_num','sms_opp_head','sms_opp_len','sms_start_time',
                    'sms_in_out']

smsTestb = pd.read_table('../data/sms_test_b.txt', header = None)
smsTestb.columns = ['uid','sms_opp_num','sms_opp_head','sms_opp_len','sms_start_time',
                    'sms_in_out']

# 对时间进行处理
smsTrain.sms_start_time = smsTrain.sms_start_time.apply(lambda x: str(x).zfill(8))
smsTest.sms_start_time = smsTest.sms_start_time.apply(lambda x: str(x).zfill(8))
smsTestb.sms_start_time = smsTestb.sms_start_time.apply(lambda x: str(x).zfill(8))

smsData = pd.concat([smsTrain, smsTest])
smsData = pd.concat([smsData, smsTestb])

# 假设第一天是周5
weekList = [5,6,7,1,2,3,4]*7
weekList = weekList[:45]

smsData['day'] = smsData.sms_start_time.map(lambda s: int(s[0:2]))
smsData['weekday'] = smsData.day.apply(lambda x: weekList[x-1] if x>0 else 6)
smsData['weeknum'] = smsData.day.apply(lambda x: (x//7) +1)

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('../data/uid_train.txt', header = None)
uidTrain.columns = ['uid', 'label']

uidTest = pd.DataFrame()
uidTest['uid'] = range(5000,10000)
uidTest.uid = uidTest.uid.apply(lambda x: 'u'+str(x).zfill(4))

feature = pd.concat([uidTrain.drop('label', axis=1), uidTest])

def get_user_sms_num_at_hour(smsData, in_out):
    if in_out not in [0,1]:
        t = smsData[['uid','sms_start_time']]
    else:
        t = smsData[smsData.sms_in_out == in_out][['uid','sms_start_time']]
    t['sms_start_time'] = t['sms_start_time'].map(lambda s: int(s[2:4]))
    
    for hour in range(24):
        s = 'user_sms_%s_num_at_hour_%s' % (str(in_out), str(hour))
        t0 = t[t.sms_start_time == hour][['uid']]
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = t.merge(t0, on='uid', how='left')
        
    t.drop(['sms_start_time'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_sms_%s_num_at_hour_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t    

def get_user_sms_num_at_day(smsData, in_out):
    if in_out not in [0,1]:
        t = smsData[['uid','day']]
    else:
        t = smsData[smsData.sms_in_out == in_out][['uid','day']]
    
    for day in range(1,46):
        s = 'user_sms_%s_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.day == day][['uid']]
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = t.merge(t0, on='uid', how='left')
        
    t.drop(['day'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_sms_%s_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t        

def get_user_sms_different_people_num_at_day(smsData, in_out):
    if in_out not in [0,1]:
        t = smsData[['uid','sms_opp_num','day']]
    else:
        t = smsData[smsData.sms_in_out == in_out][['uid','sms_opp_num','day']]
    
    t = t.groupby(['uid','day'])['sms_opp_num'].nunique().reset_index()
    
    for day in range(1,46):
        s = 'user_sms_%s_different_people_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.day == day][['uid','sms_opp_num']]
        t0.rename(columns={'sms_opp_num':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
        
    t.drop(['day','sms_opp_num'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_sms_%s_different_people_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t     

def fun4(s):    
    return int(s[0:2])*24*60*60 + int(s[2:4])*60*60 + int(s[4:6])*60 + int(s[6:8])

smsData['sms_start_time_sec'] = smsData['sms_start_time'].map(fun4).astype('str')

def fun5(s, cate):
    x = [int(i) for i in s.split(':')]
    if len(x) > 1:
        x = sorted(x)
        if cate == 0:# min
            return min(np.ediff1d(x))
        elif cate == 1: # max
            return max(np.ediff1d(x))
        elif cate == 2: # mean
            return np.mean(np.ediff1d(x))
        elif cate == 3: # median
            return np.median(np.ediff1d(x))
        elif cate == 4: # std
            return np.std(np.ediff1d(x))
        elif cate == 5: # var
            return np.var(np.ediff1d(x))
        elif cate == 6: # zeros
            return len(x) - np.count_nonzero(np.ediff1d(x))
    else:
        return -1

def make_user_sms_feature(smsData, feature):
    # user_sms_out_num
    t0 = smsData[smsData.sms_in_out == 0][['uid']]
    t0['user_sms_out_num'] = 1
    t0 = t0.groupby('uid')['user_sms_out_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_in_num
    t0 = smsData[smsData.sms_in_out == 1][['uid']]
    t0['user_sms_in_num'] = 1
    t0 = t0.groupby('uid')['user_sms_in_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_out_in_num_ratio
    feature['user_sms_out_in_num_ratio'] = (feature['user_sms_out_num'].fillna(0) + 0.001) /\
        (feature['user_sms_in_num'].fillna(0) + 0.001)
    
    # user_sms_num
    feature['user_sms_num'] = feature['user_sms_out_num'].fillna(0) + feature['user_sms_in_num'].fillna(0)
    
    # user_sms_out_ratio
    feature['user_sms_out_ratio'] = (feature['user_sms_out_num'].fillna(0) + 0.001)/\
        (feature['user_sms_num'].fillna(0) + 0.001)
    
    # user_sms_in_ratio
    feature['user_sms_in_ratio'] = (feature['user_sms_in_num'].fillna(0) + 0.001)/\
        (feature['user_sms_num'].fillna(0) + 0.001)
    
    # user_sms_opp_num_hot_value
    t0 = smsData[['sms_opp_num']]
    t0['sms_opp_num_hot_value'] = 1
    t0 = t0.groupby('sms_opp_num')['sms_opp_num_hot_value'].sum().reset_index()
    t0['sms_opp_num_hot_value'] = t0['sms_opp_num_hot_value'].astype('float')/\
        t0['sms_opp_num_hot_value'].sum()
    smsData = smsData.merge(t0, on='sms_opp_num', how='left')
    
    t0 = smsData[['uid','sms_opp_num_hot_value', 'sms_opp_num']]
    t0.drop_duplicates(['uid','sms_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['sms_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_num_hot_value':'user_sms_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_out_opp_num_hot_value
    t0 = smsData[smsData.sms_in_out == 0][['uid','sms_opp_num_hot_value', 'sms_opp_num']]
    t0.drop_duplicates(['uid','sms_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['sms_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_num_hot_value':'user_sms_out_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_in_opp_num_hot_value
    t0 = smsData[smsData.sms_in_out == 1][['uid','sms_opp_num_hot_value', 'sms_opp_num']]
    t0.drop_duplicates(['uid','sms_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['sms_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_num_hot_value':'user_sms_in_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_max_num_one_opp_num
    t0 = smsData.groupby(['uid','sms_opp_num'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].max().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_max_num_one_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_sms_max_day_num_one_opp_num
    t0 = smsData.groupby(['uid','sms_opp_num'])['day'].nunique().reset_index()
    t0 = t0.groupby('uid')['day'].max().reset_index().\
        rename(columns={'day':'user_sms_max_day_num_one_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_different_people_num
    t0 = smsData[['uid', 'sms_opp_num']]
    t0 = t0.groupby('uid')['sms_opp_num'].nunique().reset_index().\
        rename(columns={'sms_opp_num':'user_sms_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_different_people_num_ratio
    feature['user_sms_different_people_num_ratio'] = (feature['user_sms_different_people_num'].fillna(0) + 0.001)/\
        (feature['user_sms_num'].fillna(0) + 0.001)
    
    # user_sms_out_different_people_num
    t0 = smsData[smsData.sms_in_out == 0][['uid', 'sms_opp_num']]
    t0 = t0.groupby('uid')['sms_opp_num'].nunique().reset_index().\
        rename(columns={'sms_opp_num':'user_sms_out_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_in_different_people_num
    t0 = smsData[smsData.sms_in_out == 1][['uid', 'sms_opp_num']]
    t0 = t0.groupby('uid')['sms_opp_num'].nunique().reset_index().\
        rename(columns={'sms_opp_num':'user_sms_in_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
      
    # user_sms_out_in_different_people_num_ratio
    feature['user_sms_out_in_different_people_num_ratio'] = (feature['user_sms_out_different_people_num'].fillna(0) + 0.001)/\
        (feature['user_sms_in_different_people_num'].fillna(0) + 0.001)
    
    # user_sms_opp_head_hot_value
    t0 = smsData[['sms_opp_head']]
    t0['sms_opp_head_hot_value'] = 1
    t0 = t0.groupby('sms_opp_head')['sms_opp_head_hot_value'].sum().reset_index()
    t0['sms_opp_head_hot_value'] = t0['sms_opp_head_hot_value'].astype('float')/t0['sms_opp_head_hot_value'].sum()
    smsData = smsData.merge(t0, on='sms_opp_head', how='left')
    
    t0 = smsData[['uid','sms_opp_head_hot_value', 'sms_opp_head']]
    t0.drop_duplicates(['uid','sms_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['sms_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_head_hot_value':'user_sms_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_out_opp_head_hot_value
    t0 = smsData[smsData.sms_in_out == 0][['uid','sms_opp_head_hot_value','sms_opp_head']]
    t0.drop_duplicates(['uid','sms_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['sms_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_head_hot_value':'user_sms_out_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_in_opp_head_hot_value
    t0 = smsData[smsData.sms_in_out == 1][['uid','sms_opp_head_hot_value','sms_opp_head']]
    t0.drop_duplicates(['uid','sms_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['sms_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_head_hot_value':'user_sms_in_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_head_cate_num
    t0 = smsData[['uid','sms_opp_head']]
    t0 = t0.groupby('uid')['sms_opp_head'].nunique().reset_index().\
        rename(columns={'sms_opp_head':'user_sms_opp_head_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_head_0_num
    t0 = smsData[smsData.sms_opp_head == 0][['uid']]
    t0['user_sms_opp_head_0_num'] = 1
    t0 = t0.groupby('uid')['user_sms_opp_head_0_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_len_hot_value
    t0 = smsData[['sms_opp_len']]
    t0['sms_opp_len_hot_value'] = 1
    t0 = t0.groupby('sms_opp_len')['sms_opp_len_hot_value'].sum().reset_index()
    t0['sms_opp_len_hot_value'] = t0['sms_opp_len_hot_value'].astype('float')/t0['sms_opp_len_hot_value'].sum()
    smsData = smsData.merge(t0, on='sms_opp_len', how='left')
    
    t0 = smsData.groupby('uid')['sms_opp_len_hot_value'].sum().reset_index().\
        rename(columns={'sms_opp_len_hot_value':'user_sms_opp_len_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_len_11
    t0 = smsData[smsData.sms_opp_len == 11][['uid']]
    t0['user_sms_opp_len_11'] = 1
    t0 = t0.groupby('uid')['user_sms_opp_len_11'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_len_13
    t0 = smsData[smsData.sms_opp_len == 13][['uid']]
    t0['user_sms_opp_len_13'] = 1
    t0 = t0.groupby('uid')['user_sms_opp_len_13'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_len_9
    t0 = smsData[smsData.sms_opp_len == 9][['uid']]
    t0['user_sms_opp_len_9'] = 1
    t0 = t0.groupby('uid')['user_sms_opp_len_9'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_opp_len_other
    t0 = smsData[(smsData.sms_opp_len != 9)&(smsData.sms_opp_len != 11)&(smsData.sms_opp_len != 13)][['uid']]
    t0['user_sms_opp_len_other'] = 1
    t0 = t0.groupby('uid')['user_sms_opp_len_other'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
      
    # user_sms_hour_num
    t0 = smsData[['uid','sms_start_time']]
    t0['sms_start_time'] = t0['sms_start_time'].map(lambda s: int(s[2:4]))
    t0 = t0.groupby('uid')['sms_start_time'].nunique().reset_index().\
        rename(columns={'sms_start_time':'user_sms_hour_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_day_num
    t0 = smsData.groupby('uid')['day'].nunique().reset_index().\
        rename(columns={'day':'user_sms_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_in_day_num
    t0 = smsData[smsData.sms_in_out == 1].groupby('uid')['day'].nunique().reset_index().\
        rename(columns={'day':'user_sms_in_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_in_day_num_ratio
    feature['user_sms_in_day_num_ratio'] = (feature['user_sms_in_day_num'].fillna(0) + 0.001)/\
        (feature['user_sms_day_num'].fillna(0) + 0.001)
        
    # user_sms_max_num_one_day
    t0 = smsData.groupby(['uid','day'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].max().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_max_num_one_day'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_out_max_num_one_day
    t0 = smsData[smsData.sms_in_out == 0].groupby(['uid','day'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].max().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_out_max_num_one_day'})
    feature = feature.merge(t0, on='uid', how='left') 

    # user_sms_in_max_num_one_day
    t0 = smsData[smsData.sms_in_out == 1].groupby(['uid','day'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].max().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_in_max_num_one_day'})
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_sms_max_num_one_time
    t0 = smsData.groupby(['uid','sms_start_time'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].max().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_max_num_one_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_max_num_one_hour
    t0 = smsData[['uid','sms_start_time']]
    t0['sms_start_time'] = t0['sms_start_time'].map(lambda s: s[0:4])
    t0['sms_max_num_one_hour'] = 1
    t0 = t0.groupby(['uid','sms_start_time'])['sms_max_num_one_hour'].sum().reset_index()
    t0 = t0.groupby('uid')['sms_max_num_one_hour'].max().reset_index().\
        rename(columns={'sms_max_num_one_hour':'user_sms_max_num_one_hour'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_max_num_one_opp_head
    t0 = smsData.groupby(['uid','sms_opp_head'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].max().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_max_num_one_opp_head'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_min_num_one_opp_head
    t0 = smsData.groupby(['uid','sms_opp_head'])['sms_opp_len'].count().reset_index()
    t0 = t0.groupby('uid')['sms_opp_len'].min().reset_index().\
        rename(columns={'sms_opp_len':'user_sms_min_num_one_opp_head'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_x_num_at_hour_y
    t0 = get_user_sms_num_at_hour(smsData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_x_num_at_day_y
    t0 = get_user_sms_num_at_day(smsData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_sms_x_different_people_num_at_day_y
    t0 = get_user_sms_different_people_num_at_day(smsData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    
    # 45天内通话间隔统计特征
    t = smsData.groupby('uid')['sms_start_time_sec'].agg(lambda x: ':'.join(x)).reset_index()
    
    t['user_sms_during_min'] = t['sms_start_time_sec'].map(lambda s: fun5(s, 0))
    t['user_sms_during_max'] = t['sms_start_time_sec'].map(lambda s: fun5(s, 1))
    t['user_sms_during_mean'] = t['sms_start_time_sec'].map(lambda s: fun5(s, 2))    
    t['user_sms_during_median'] = t['sms_start_time_sec'].map(lambda s: fun5(s, 3))
    t['user_sms_during_std'] = t['sms_start_time_sec'].map(lambda s: fun5(s, 4))
    t['user_sms_during_zero'] = t['sms_start_time_sec'].map(lambda s: fun5(s, 6))
    
    t.drop(['sms_start_time_sec'], axis=1, inplace=True)

    feature = feature.merge(t, on='uid', how='left')
    
    
    return feature


# 提取特征
feature = make_user_sms_feature(smsData, feature)

# feature.to_csv('../data/feature_sms_03b.csv', index=False)

# 训练集
train = feature[:4999].copy()
train = train.merge(uidTrain, on='uid', how='left')

# 打乱顺序
np.random.seed(201805)
idx = np.random.permutation(len(train))
train = train.iloc[idx]

X_train = train.drop(['uid','label'], axis=1).values
y_train = train.label.values

# 测试集
test = feature[6999:].copy()

X_test = test.drop(['uid'], axis=1).values

"""
lgb = lightgbm.LGBMClassifier(boosting_type='gbdt', 
          objective= 'binary',
          metric= 'auc',
          min_child_weight= 1.5,
          num_leaves = 2**5,
          lambda_l2= 10,
          subsample= 0.7,
          colsample_bytree= 0.5,
          colsample_bylevel= 0.5,
          learning_rate= 0.1,
          scale_pos_weight= 20,
          seed= 201805,
          nthread= 4,
          silent= True)

F1: 0.586988304094 AUC: 0.904528746849 Score: 0.777512569747
             precision    recall  f1-score   support

          0       0.97      0.75      0.84      4099
          1       0.44      0.89      0.59       900

avg / total       0.87      0.77      0.80      4999
"""

lgb = lightgbm.LGBMClassifier(random_state=201805)
"""
F1: 0.580025608195 AUC: 0.907820335583 Score: 0.776702444628
             precision    recall  f1-score   support

          0       0.90      0.95      0.92      4099
          1       0.68      0.50      0.58       900

avg / total       0.86      0.87      0.86      4999
"""

def fitModel(model, feature1):
    X = feature1.drop(['uid','label'], axis=1).values
    y = feature1.label.values
    
    #lgb_y_pred = cross_val_predict(model, X, y, cv=5,
    #                       verbose=2, method='predict')
    lgb_y_proba = cross_val_predict(model, X, y, cv=5,
                                verbose=2, method='predict_proba')[:,1]
    
    lgb_y_pred = (lgb_y_proba >= 0.50)*1
    f1score = f1_score(y, lgb_y_pred)
    aucscore = roc_auc_score(y, lgb_y_proba)
    print('F1:', f1score,
          'AUC:', aucscore,
          'Score:', f1score*0.4 + aucscore*0.6,
          'Num:', sum(lgb_y_pred))
    print(classification_report(y, lgb_y_pred))
    
    model.fit(X, y)

    featureList0 = list(feature1.drop(['uid','label'], axis=1))
    featureImportant = pd.DataFrame()
    featureImportant['feature'] = featureList0
    featureImportant['score'] = model.feature_importances_
    featureImportant.sort_values(by='score', ascending=False, inplace=True)
    featureImportant.reset_index(drop=True, inplace=True)
    print(featureImportant)
    


# 交叉验证模型
fitModel(lgb, train)
















