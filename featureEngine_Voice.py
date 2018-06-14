# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:02:29 2018

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

# voice_train.txt 用户通话记录数据
voiceTrain = pd.read_table('../data/voice_train.txt', header = None)
voiceTrain.columns = ['uid', 'call_opp_num', 'call_opp_head', 'call_opp_len', 'call_start_time',
                      'call_end_time', 'call_type', 'call_in_out']

voiceTest = pd.read_table('../data/voice_test_a.txt', header = None)
voiceTest.columns = ['uid', 'call_opp_num', 'call_opp_head', 'call_opp_len', 'call_start_time',
                      'call_end_time', 'call_type', 'call_in_out']

voiceTrain.call_start_time = voiceTrain.call_start_time.apply(lambda x: str(x).zfill(8))
voiceTrain.call_end_time = voiceTrain.call_end_time.apply(lambda x: str(x).zfill(8))
voiceTest.call_start_time = voiceTest.call_start_time.apply(lambda x: str(x).zfill(8))
voiceTest.call_end_time = voiceTest.call_end_time.apply(lambda x: str(x).zfill(8))


# 假设第一天是周5
weekList = [5,6,7,1,2,3,4]*7
weekList = weekList[:45]

voiceData = pd.concat([voiceTrain, voiceTest])

# 计算时间差
def fun1(s):
    date1, date2 = s.split(':')
    if (len(date1)==8) & (len(date2)==8):
        date1_sec = int(date1[0:2])*24*60*60 + int(date1[2:4])*60*60 + int(date1[4:6])*60 + int(date1[6:8])
        date2_sec = int(date2[0:2])*24*60*60 + int(date2[2:4])*60*60 + int(date2[4:6])*60 + int(date2[6:8])
        return (date2_sec - date1_sec)/60.0
    else:
        return 0
    
# 计算时间差
def fun2(s):
    date1, date2 = s.split(':')
    if (len(date1)==8) & (len(date2)==8):
        date1_sec = int(date1[0:2])*24*60*60 + int(date1[2:4])*60*60 + int(date1[4:6])*60 + int(date1[6:8])
        date2_sec = int(date2[0:2])*24*60*60 + int(date2[2:4])*60*60 + int(date2[4:6])*60 + int(date2[6:8])
        return (date2_sec - date1_sec)
    else:
        return 0

# 每次通话时长   
voiceData['call_period_time'] = voiceData.call_start_time + ":" + voiceData.call_end_time
voiceData.call_period_time = voiceData.call_period_time.map(fun2)

def get_user_call_num_at_day(voiceData, in_out):
    if in_out == 2:
        t = voiceData[['uid', 'call_start_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time']]
    t['call_start_time'] = t['call_start_time'].map(lambda s: int(s[0:2]))
    
    for day in range(1, 46):
        s = 'user_call_%s_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.call_start_time == day][['uid']]
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    for day in range(1, 46):
        s = 'user_call_%s_num_at_day_%s' % (str(in_out), str(day))
        t.drop(s, axis=1, inplace=True)
    
    return t

def get_user_call_num_at_hour(voiceData, in_out):
    if in_out == 2:
        t = voiceData[['uid', 'call_start_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time']]
    t['call_start_time'] = t['call_start_time'].map(lambda s: int(s[2:4]))
    
    for hour in range(24):
        s = 'user_call_%s_num_at_hour_%s' % (str(in_out), str(hour))
        t0 = t[t.call_start_time == hour][['uid']]
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_num_at_hour_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    for day in range(5):
        s = 'user_call_%s_num_at_hour_%s' % (str(in_out), str(day))
        t.drop(s, axis=1, inplace=True)
        
    return t

def get_user_call_hour_num_at_day(voiceData, in_out):
    if in_out == 2:
        t = voiceData[['uid', 'call_start_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time']]
    t['day'] = t['call_start_time'].map(lambda s: int(s[0:2]))
    t['hour'] = t['call_start_time'].map(lambda s: int(s[2:4]))
    
    for day in range(1, 46):
        s = 'user_call_%s_hour_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.day == day][['uid','hour']]
        t0 = t0.groupby('uid')['hour'].nunique().reset_index().\
            rename(columns={'hour':s})
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time','day','hour'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_hour_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    for day in range(1, 46):
        s = 'user_call_%s_hour_num_at_day_%s' % (str(in_out), str(day))
        t.drop(s, axis=1, inplace=True)
    
    return t

def get_user_call_different_people_num_at_day(voiceData, in_out):
    if in_out == 2:
        t = voiceData[['uid','call_opp_num', 'call_start_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_opp_num','call_start_time']]
        
    t['call_start_time'] = t['call_start_time'].map(lambda s: int(s[0:2]))
    
    for day in range(1, 46):
        s = 'user_call_%s_different_people_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.call_start_time == day][['uid','call_opp_num']]
        t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
            rename(columns={'call_opp_num':s})
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time','call_opp_num'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_different_people_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    for day in range(1, 46):
        s = 'user_call_%s_different_people_num_at_day_%s' % (str(in_out), str(day))
        t.drop(s, axis=1, inplace=True)
    
    return t

def get_user_call_opp_num_hot_value_at_day(voiceData, in_out):
    
    if 'call_opp_num_hot_value' in list(voiceData):
        voiceData.drop('call_opp_num_hot_value', axis=1, inplace=True)
    
    t0 = voiceData[['call_opp_num']]
    t0['call_opp_num_hot_value'] = 1
    t0 = t0.groupby('call_opp_num')['call_opp_num_hot_value'].sum().reset_index()
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    
    if in_out == 2:
        t = voiceData[['uid','call_start_time','call_opp_num_hot_value']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time','call_opp_num_hot_value']]
    
    t['call_start_time'] = t['call_start_time'].map(lambda s: int(s[0:2]))
    
    
    for day in range(1, 46):
        s = 'user_call_%s_opp_num_hot_value_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.call_start_time == day][['uid','call_opp_num_hot_value']]
        t0 = t0.groupby('uid')['call_opp_num_hot_value'].sum().reset_index().\
            rename(columns={'call_opp_num_hot_value':s})
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time', 'call_opp_num_hot_value'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_opp_num_hot_value_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    #for day in range(1, 46):
    #    s = 'user_call_%s_opp_num_hot_value_at_day_%s' % (str(in_out), str(day))
    #    t.drop(s, axis=1, inplace=True)
    
    return t


def get_user_call_time_at_day(voiceData, in_out):
    if in_out == 2:
        t = voiceData[['uid', 'call_start_time', 'call_end_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time','call_end_time']]
    t['day'] = t['call_start_time'].map(lambda s: int(s[0:2]))
    t['call_start_time'] = t['call_start_time']+':'+t['call_end_time']
    t['call_start_time'] = t['call_start_time'].apply(fun1)
    
    for day in range(1, 46):
        s = 'user_call_%s_time_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.day == day][['uid','call_start_time']]
        t0 = t0.groupby('uid')['call_start_time'].sum().reset_index().\
            rename(columns={'call_start_time':s})
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time','day', 'call_end_time'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_time_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    #for day in range(1, 46):
    #    s = 'user_call_%s_time_at_day_%s' % (str(in_out), str(day))
    #    t.drop(s, axis=1, inplace=True)
    
    return t

def get_user_call_opp_len_12_at_day(voiceData, in_out):
    if in_out == 2:
        t = voiceData[['uid','call_opp_len', 'call_start_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time','call_opp_len']]
    t['call_start_time'] = t['call_start_time'].map(lambda s: int(s[0:2]))
    
    for day in range(1, 46):
        s = 'user_call_%s_opp_len_12_at_day_%s' % (str(in_out), str(day))
        t0 = t[(t.call_start_time == day)&(t.call_opp_len == 12)][['uid']]
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['call_start_time', 'call_opp_len'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_opp_len_12_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    #for day in range(1, 46):
    #    s = 'user_call_%s_time_at_day_%s' % (str(in_out), str(day))
    #    t.drop(s, axis=1, inplace=True)
    
    return t

# 用户耀日通话次数
def get_user_different_weekday_call_nums(voiceData):
    t = voiceData[['uid','call_start_time']]
    t['call_start_time'] = t['call_start_time'].apply(lambda x: int(x[0:2]))
    t['call_start_time'] = t['call_start_time'].apply(lambda x: weekList[x-1] if x>0 else 0)
    for weekday in range(1,8):
        t0 = t[t.call_start_time == weekday][['uid']]
        s = 'user_call_nums_at_weekday_%s' % weekday
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = pd.merge(t, t0, on='uid', how='left')
        
    t.fillna(0, inplace=True)
    t.drop(['call_start_time'], axis=1, inplace=True)
    t.drop_duplicates(inplace=True)
    
    # 复制一下
    df = t.copy()
    
    # max
    df1 = df.drop('uid', axis=1).apply(np.max, axis=1)
    t['user_call_nums_at_weekday_max'] = df1.values
    # min
    df1 = df.drop('uid', axis=1).apply(np.min, axis=1)
    t['user_call_nums_at_weekday_min'] = df1.values
    # median
    #df1 = df.drop('uid', axis=1).apply(np.median, axis=1)
    #t['user_call_nums_at_weekday_median'] = df1.values
    # mean
    df1 = df.drop('uid', axis=1).apply(np.mean, axis=1)
    t['user_call_nums_at_weekday_mean'] = df1.values
    # std
    df1 = df.drop('uid', axis=1).apply(np.std, axis=1)
    t['user_call_nums_at_weekday_std'] = df1.values
    # sum
    #df1 = df.drop('uid', axis=1).apply(np.sum, axis=1)
    #t['user_call_nums_at_weekday_sum'] = df1.values
    
    return t 

def make_user_voice_feature(voiceData, feature):
    # user_call_opp_num_hot_value
    t0 = voiceData[['call_opp_num']]
    t0['call_opp_num_hot_value'] = 1
    t0 = t0.groupby('call_opp_num')['call_opp_num_hot_value'].sum().reset_index()
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    
    t0 = voiceData[['uid','call_opp_num', 'call_opp_num_hot_value']]
    t0.drop_duplicates(['uid','call_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_num_hot_value':'user_call_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_num_hot_value
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_opp_num', 'call_opp_num_hot_value']]
    t0.drop_duplicates(['uid','call_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_num_hot_value':'user_call_out_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')  
    
    # user_call_in_opp_num_hot_value
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_opp_num', 'call_opp_num_hot_value']]
    t0.drop_duplicates(['uid','call_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_num_hot_value':'user_call_in_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')     
    
    feature['user_call_in_opp_num_hot_value_ratio'] = feature['user_call_in_opp_num_hot_value'].fillna(0)/\
        (feature['user_call_opp_num_hot_value'].fillna(0) + 0.1)
    
    # user_call_different_people_num
    t0 = voiceData[['uid','call_opp_num']]
    t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_different_people_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_opp_num']]
    t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_out_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_different_people_num
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_opp_num']]
    t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_in_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')  
    
    # user_call_in_different_people_num_ratio
    feature['user_call_in_different_people_num_ratio'] = feature['user_call_in_different_people_num'].fillna(0)/\
        (feature['user_call_different_people_num'].fillna(0) + 0.1)
    
    # user_call_num
    t0 = voiceData[['uid']]
    t0['user_call_num'] = 1
    t0 = t0.groupby('uid')['user_call_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid']]
    t0['user_call_out_num'] = 1
    t0 = t0.groupby('uid')['user_call_out_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_in_num
    #t0 = voiceData[voiceData.call_in_out == 1][['uid']]
    #t0['user_call_in_num'] = 1
    #t0 = t0.groupby('uid')['user_call_in_num'].sum().reset_index()
    #feature = feature.merge(t0, on='uid', how='left')

    # user_call_out_num_ratio
    #feature['user_call_out_num_ratio'] = feature['user_call_out_num'].fillna(0)/\
    #    (feature['user_call_num'].fillna(0) + 0.1)
    
    # user_call_opp_head_hot_value
    t0 = voiceData[['call_opp_head']]
    t0['call_opp_head_hot_value'] = 1
    t0 = t0.groupby('call_opp_head')['call_opp_head_hot_value'].sum().reset_index()
    voiceData = voiceData.merge(t0, on='call_opp_head', how='left')
    
    t0 = voiceData[['uid', 'call_opp_head', 'call_opp_head_hot_value']]
    t0.drop_duplicates(['uid','call_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_head_hot_value
    t0 = voiceData[voiceData.call_in_out == 0][['uid', 'call_opp_head', 'call_opp_head_hot_value']]
    t0.drop_duplicates(['uid','call_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_out_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')   
    
    # user_call_in_opp_head_hot_value
    t0 = voiceData[voiceData.call_in_out == 1][['uid', 'call_opp_head', 'call_opp_head_hot_value']]
    t0.drop_duplicates(['uid','call_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_in_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')   
    
    # user_call_opp_head_cate_num
    t0 = voiceData[['uid', 'call_opp_head']]
    t0 = t0.groupby('uid')['call_opp_head'].nunique().reset_index().\
        rename(columns={'call_opp_head':'user_call_opp_head_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_head_cate_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid', 'call_opp_head']]
    t0 = t0.groupby('uid')['call_opp_head'].nunique().reset_index().\
        rename(columns={'call_opp_head':'user_call_out_opp_head_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_day_num
    t0 = voiceData[['uid','call_start_time']]
    t0['call_start_time'] = t0['call_start_time'].map(lambda s: int(s[0:2]))
    t0 = t0.groupby('uid')['call_start_time'].nunique().reset_index().\
        rename(columns={'call_start_time':'user_call_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_day_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_start_time']]
    t0['call_start_time'] = t0['call_start_time'].map(lambda s: int(s[0:2]))
    t0 = t0.groupby('uid')['call_start_time'].nunique().reset_index().\
        rename(columns={'call_start_time':'user_call_out_day_num'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_in_day_num
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_start_time']]
    t0['call_start_time'] = t0['call_start_time'].map(lambda s: int(s[0:2]))
    t0 = t0.groupby('uid')['call_start_time'].nunique().reset_index().\
        rename(columns={'call_start_time':'user_call_in_day_num'})
    feature = feature.merge(t0, on='uid', how='left')   
    
    #feature['user_call_out_in_day_num_ratio'] = feature['user_call_out_day_num'].fillna(0) /\
    #    (feature['user_call_in_day_num'].fillna(0) + 0.1)
        
    # user_call_out_total_time
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_start_time','call_end_time']]
    t0['start_end_time'] = t0['call_start_time'] + ':' + t0['call_end_time']
    t0['start_end_time'] = t0['start_end_time'].map(fun1)
    t0 = t0.groupby('uid')['start_end_time'].sum().reset_index().\
        rename(columns={'start_end_time':'user_call_out_total_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_total_time
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_start_time','call_end_time']]
    t0['start_end_time'] = t0['call_start_time'] + ':' + t0['call_end_time']
    t0['start_end_time'] = t0['start_end_time'].map(fun1)
    t0 = t0.groupby('uid')['start_end_time'].sum().reset_index().\
        rename(columns={'start_end_time':'user_call_in_total_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    feature['user_call_total_time'] = feature['user_call_out_total_time'].fillna(0) +\
        feature['user_call_in_total_time'].fillna(0)
     
    # user_call_type_1
    t0 = voiceData[voiceData.call_type == 1][['uid']]
    t0['user_call_type_1'] = 1
    t0 = t0.groupby('uid')['user_call_type_1'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')

    # user_call_type_3
    #t0 = voiceData[voiceData.call_type == 3][['uid']]
    #t0['user_call_type_3'] = 1
    #t0 = t0.groupby('uid')['user_call_type_3'].sum().reset_index()
    
    #feature = feature.merge(t0, on='uid', how='left')        
    
    # user_call_type_others(2,4,5)
    t0 = voiceData[voiceData.call_type.isin([2,4,5])][['uid']]
    t0['user_call_type_other'] = 1
    t0 = t0.groupby('uid')['user_call_type_other'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_opp_len_11
    t0 = voiceData[voiceData.call_opp_len == 11][['uid']]
    t0['user_call_opp_len_11'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_11'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_len_12
    t0 = voiceData[voiceData.call_opp_len == 12][['uid']]
    t0['user_call_opp_len_12'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_12'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')

    # user_call_opp_len_5
    t0 = voiceData[voiceData.call_opp_len == 5][['uid']]
    t0['user_call_opp_len_5'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_5'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_len_other
    t0 = voiceData[(voiceData.call_opp_len != 5)&(voiceData.call_opp_len != 11)&(voiceData.call_opp_len != 12)][['uid']]
    t0['user_call_opp_len_other'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_other'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')  
    
    # user_call_other_max_num
    #t0 = voiceData[voiceData.call_in_out == 0][['uid','call_opp_num']]
    #t0['max_num'] = 1
    #t0 = t0.groupby(['uid','call_opp_num'])['max_num'].sum().reset_index()
    #t0 = t0.groupby('uid')['max_num'].max().reset_index().\
    #    rename(columns={'max_num':'user_call_other_max_num'})
    #feature = feature.merge(t0, on='uid', how='left')

    # user_call_mean_time
    feature['user_call_mean_time'] = feature['user_call_total_time'].fillna(0)/\
        (feature['user_call_num'].fillna(0) + 0.1)
        
    
    
    """
    # user_call_out_call_type_1
    t = voiceData[voiceData.call_in_out == 0][['uid','call_type']]
    t0 = t[t.call_type == 1]
    t0['user_call_out_call_type_1'] = 1
    t0 = t0.groupby('uid')['user_call_out_call_type_1'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')
    # user_call_out_call_type_other
    t = voiceData[voiceData.call_in_out == 0][['uid','call_type']]
    t0 = t[t.call_type != 1]
    t0['user_call_out_call_type_other'] = 1
    t0 = t0.groupby('uid')['user_call_out_call_type_other'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid']]
    t0['user_call_out_num'] = 1
    t0 = t0.groupby('uid')['user_call_out_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_num
    t0 = voiceData[voiceData.call_in_out == 1][['uid']]
    t0['user_call_in_num'] = 1
    t0 = t0.groupby('uid')['user_call_in_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_in_ratio
    feature['user_call_out_in_ratio'] = feature['user_call_out_num'].fillna(0) /\
        (feature['user_call_in_num'].fillna(0) + 0.1)
    
    # user_call_out_day_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_start_time']]
    t0['call_start_time'] = t0['call_start_time'].map(lambda s: int(s[0:2]))
    t0 = t0.groupby('uid')['call_start_time'].nunique().reset_index().\
        rename(columns={'call_start_time':'user_call_out_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_day_num
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_start_time']]
    t0['call_start_time'] = t0['call_start_time'].map(lambda s: int(s[0:2]))
    t0 = t0.groupby('uid')['call_start_time'].nunique().reset_index().\
        rename(columns={'call_start_time':'user_call_in_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_in_day_num_ratio
    feature['user_call_out_in_day_num_ratio'] = feature['user_call_out_day_num'].fillna(0) /\
        (feature['user_call_in_day_num'].fillna(0) + 0.1)
    
    # user_call_out_total_time
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_start_time','call_end_time']]
    t0['start_end_time'] = t0['call_start_time'] + ':' + t0['call_end_time']
    t0['start_end_time'] = t0['start_end_time'].map(fun1)
    t0 = t0.groupby('uid')['start_end_time'].sum().reset_index().\
        rename(columns={'start_end_time':'user_call_out_total_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_total_time
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_start_time','call_end_time']]
    t0['start_end_time'] = t0['call_start_time'] + ':' + t0['call_end_time']
    t0['start_end_time'] = t0['start_end_time'].map(fun1)
    t0 = t0.groupby('uid')['start_end_time'].sum().reset_index().\
        rename(columns={'start_end_time':'user_call_in_total_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_in_total_time_ratio
    feature['user_call_out_in_total_time_ratio'] = feature['user_call_out_total_time'].fillna(0)/\
        (feature['user_call_in_total_time'].fillna(0) + 0.1)
    
    # user_call_total_time
    feature['user_call_total_time'] = feature['user_call_out_total_time'].fillna(0) +\
        feature['user_call_in_total_time'].fillna(0)
    
    # user_call_different_people_num
    t0 = voiceData[['uid','call_opp_num']]
    t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_different_people_num
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_opp_num']]
    t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_out_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_different_people_num
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_opp_num']]
    t0 = t0.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_in_different_people_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_in_different_people_num
    feature['user_call_out_in_different_people_num_ratio'] = feature['user_call_out_different_people_num'].fillna(0)/\
        (feature['user_call_in_different_people_num'].fillna(0) + 0.1)
    
    # user_call_total_num
    feature['user_call_total_num'] = feature['user_call_out_num'].fillna(0) + \
        feature['user_call_in_num'].fillna(0)
    
    # user_call_type_1
    t0 = voiceData[voiceData.call_type == 1][['uid']]
    t0['user_call_type_1'] = 1
    t0 = t0.groupby('uid')['user_call_type_1'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_type_3
    t0 = voiceData[voiceData.call_type == 3][['uid']]
    t0['user_call_type_3'] = 1
    t0 = t0.groupby('uid')['user_call_type_3'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_type_others(2,4,5)
    t0 = voiceData[voiceData.call_type.isin([2,4,5])][['uid']]
    t0['user_call_type_other'] = 1
    t0 = t0.groupby('uid')['user_call_type_other'].sum().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_type_1_ratio
    feature['user_call_type_1_ratio'] = feature['user_call_type_1'].fillna(0)/\
        (feature['user_call_total_num'].fillna(0) + 0.1)
    
    # user_call_opp_len_11
    t0 = voiceData[voiceData.call_opp_len == 11][['uid']]
    t0['user_call_opp_len_11'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_11'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_len_12
    t0 = voiceData[voiceData.call_opp_len == 12][['uid']]
    t0['user_call_opp_len_12'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_12'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_len_5
    t0 = voiceData[voiceData.call_opp_len == 5][['uid']]
    t0['user_call_opp_len_5'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_5'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_len_other
    t0 = voiceData[(voiceData.call_opp_len != 5)&(voiceData.call_opp_len != 11)&(voiceData.call_opp_len != 12)][['uid']]
    t0['user_call_opp_len_other'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_other'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_len_11
    t0 = voiceData[(voiceData.call_in_out == 0)&(voiceData.call_opp_len == 11)][['uid']]
    t0['user_call_out_opp_len_11'] = 1
    t0 = t0.groupby('uid')['user_call_out_opp_len_11'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_len_12
    t0 = voiceData[(voiceData.call_in_out == 0)&(voiceData.call_opp_len == 12)][['uid']]
    t0['user_call_out_opp_len_12'] = 1
    t0 = t0.groupby('uid')['user_call_out_opp_len_12'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_len_other
    t0 = voiceData[(voiceData.call_in_out == 0)&(voiceData.call_opp_len != 12)&(voiceData.call_opp_len != 11)][['uid']]
    t0['user_call_out_opp_len_other'] = 1
    t0 = t0.groupby('uid')['user_call_out_opp_len_other'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_opp_len_11
    t0 = voiceData[(voiceData.call_in_out == 1)&(voiceData.call_opp_len == 11)][['uid']]
    t0['user_call_in_opp_len_11'] = 1
    t0 = t0.groupby('uid')['user_call_in_opp_len_11'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_opp_len_12
    t0 = voiceData[(voiceData.call_in_out == 1)&(voiceData.call_opp_len == 12)][['uid']]
    t0['user_call_in_opp_len_12'] = 1
    t0 = t0.groupby('uid')['user_call_in_opp_len_12'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_opp_len_other
    t0 = voiceData[(voiceData.call_in_out == 1)&(voiceData.call_opp_len != 11)&(voiceData.call_opp_len != 12)][['uid']]
    t0['user_call_in_opp_len_other'] = 1
    t0 = t0.groupby('uid')['user_call_in_opp_len_other'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # 
    t0 = voiceData[['call_opp_head']]
    t0['call_opp_head_hot_value'] = 1
    t0 = t0.groupby('call_opp_head')['call_opp_head_hot_value'].sum().reset_index()
    
    voiceData = voiceData.merge(t0, on='call_opp_head', how='left')
    # user_call_opp_head_hot_value
    t0 = voiceData[['uid','call_opp_head','call_opp_head_hot_value']]
    t0.drop_duplicates(subset=['uid','call_opp_head'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_head_cate_num
    t0 = voiceData[['uid','call_opp_head']]
    t0 = t0.groupby('uid')['call_opp_head'].nunique().reset_index().\
        rename(columns={'call_opp_head':'user_call_opp_head_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_num_hot_value
    t0 = voiceData[['call_opp_num']]
    t0['call_opp_num_hot_value'] = 1
    t0 = t0.groupby('call_opp_num')['call_opp_num_hot_value'].sum().reset_index()
    
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    
    t0 = voiceData[['uid','call_opp_num', 'call_opp_num_hot_value']]
    t0.drop_duplicates(subset=['uid','call_opp_num'], inplace=True)
    t0 = t0.groupby('uid')['call_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_num_hot_value':'user_call_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_num_at_day_y
    #t0 = get_user_call_num_at_day(voiceData, 2)
    #feature = feature.merge(t0, on='uid', how='left')
    
    #t0 = get_user_call_num_at_day(voiceData, 1)
    #feature = feature.merge(t0, on='uid', how='left')
    
    #t0 = get_user_call_num_at_day(voiceData, 0)
    #feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_num_at_hour_y
    t0 = get_user_call_num_at_hour(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_time_at_day_y
    t0 = get_user_call_time_at_day(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_opp_num_hot_value_at_day_y
    t0 = get_user_call_opp_num_hot_value_at_day(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    #t0 = get_user_different_weekday_call_nums(voiceData)
    #feature = feature.merge(t0, on='uid', how='left')
    # user_call_x_opp_len_12_at_day_y
    #t0 = get_user_call_opp_len_12_at_day(voiceData, 2)
    #feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_different_people_num_at_day_y
    #t0 = get_user_call_different_people_num_at_day(voiceData, 2)
    #feature = feature.merge(t0, on='uid', how='left')
    
    #t0 = get_user_call_num_at_hour(voiceData, 0)
    #feature = feature.merge(t0, on='uid', how='left')
    #t0 = get_user_call_hour_num_at_day(voiceData, 2)
    #feature = feature.merge(t0, on='uid', how='left')
    """
    
    return feature

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('../data/uid_train.txt', header = None)
uidTrain.columns = ['uid', 'label']

uidTest = pd.DataFrame()
uidTest['uid'] = range(5000,7000)
uidTest.uid = uidTest.uid.apply(lambda x: 'u'+str(x).zfill(4))

feature = pd.concat([uidTrain.drop('label', axis=1), uidTest])

# 提取特征
feature = make_user_voice_feature(voiceData, feature)

# feature.to_csv('../data/feature_voice_04.csv', index=False)

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
test = feature[4999:].copy()

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
"""
lgb = lightgbm.LGBMClassifier(random_state=201805)

def fitModel(model, feature1):
    X = feature1.drop(['uid','label'], axis=1).values
    y = feature1.label.values
    
    lgb_y_pred = cross_val_predict(model, X, y, cv=5,
                           verbose=2, method='predict')
    lgb_y_proba = cross_val_predict(model, X, y, cv=5,
                                verbose=2, method='predict_proba')[:,1]
    
    f1score = f1_score(y, lgb_y_pred)
    aucscore = roc_auc_score(y, lgb_y_proba)
    print('F1:', f1score,
          'AUC:', aucscore,
          'Score:', f1score*0.4 + aucscore*0.6)
    print(classification_report(y, lgb_y_pred))
    
    model.fit(X, y)

    featureList0 = list(feature1.drop(['uid','label'], axis=1))
    featureImportant = pd.DataFrame()
    featureImportant['feature'] = featureList0
    featureImportant['score'] = lgb.feature_importances_
    featureImportant.sort_values(by='score', ascending=False, inplace=True)
    featureImportant.reset_index(drop=True, inplace=True)
    print(featureImportant)
    


# 交叉验证模型
fitModel(lgb, train)















