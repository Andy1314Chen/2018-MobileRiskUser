# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:46:50 2018

@author: MSIK
"""

import pandas as pd
import numpy as np
from datetime import date

import warnings
warnings.filterwarnings('ignore')

import lightgbm
import xgboost
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

# voice_train.txt 用户通话记录数据
voiceTrain = pd.read_table('../data/voice_train.txt', header = None)
voiceTrain.columns = ['uid', 'call_opp_num', 'call_opp_head', 'call_opp_len', 'call_start_time',
                      'call_end_time', 'call_type', 'call_in_out']

voiceTest = pd.read_table('../data/voice_test_a.txt', header = None)
voiceTest.columns = ['uid', 'call_opp_num', 'call_opp_head', 'call_opp_len', 'call_start_time',
                      'call_end_time', 'call_type', 'call_in_out']

voiceTestb = pd.read_table('../data/voice_test_b.txt', header = None)
voiceTestb.columns = ['uid', 'call_opp_num', 'call_opp_head', 'call_opp_len', 'call_start_time',
                      'call_end_time', 'call_type', 'call_in_out']

voiceTrain.call_start_time = voiceTrain.call_start_time.apply(lambda x: str(x).zfill(8))
voiceTrain.call_end_time = voiceTrain.call_end_time.apply(lambda x: str(x).zfill(8))
voiceTest.call_start_time = voiceTest.call_start_time.apply(lambda x: str(x).zfill(8))
voiceTest.call_end_time = voiceTest.call_end_time.apply(lambda x: str(x).zfill(8))
voiceTestb.call_start_time = voiceTestb.call_start_time.apply(lambda x: str(x).zfill(8))
voiceTestb.call_end_time = voiceTestb.call_end_time.apply(lambda x: str(x).zfill(8))


# 假设第一天是周5
weekList = [5,6,7,1,2,3,4]*7
weekList = weekList[:45]

voiceData = pd.concat([voiceTrain, voiceTest])
voiceData = pd.concat([voiceData, voiceTestb])
voiceData.call_opp_head = voiceData.call_opp_head.astype('str')
voiceData['day'] = voiceData.call_start_time.map(lambda s: int(s[0:2]))
voiceData['weekday'] = voiceData.day.apply(lambda x: weekList[x-1] if x>0 else 6)
voiceData['weeknum'] = voiceData.day.apply(lambda x: (x//7) +1)

# 计算时间差
def fun1(s):
    date1, date2 = s.split(':')
    if (len(date1)==8) & (len(date2)==8):
        date1_sec = int(date1[0:2])*24*60*60 + int(date1[2:4])*60*60 + int(date1[4:6])*60 + int(date1[6:8])
        date2_sec = int(date2[0:2])*24*60*60 + int(date2[2:4])*60*60 + int(date2[4:6])*60 + int(date2[6:8])
        return (date2_sec - date1_sec)
    else:
        return 0

voiceData['call_time'] = voiceData['call_start_time'] + ':' + voiceData['call_end_time']
voiceData['call_time'] = voiceData['call_time'].map(fun1)

def make_voice_feature(voiceData, feature):
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
    t0 = voiceData[voiceData.call_in_out == 1][['uid']]
    t0['user_call_in_num'] = 1
    t0 = t0.groupby('uid')['user_call_in_num'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_num_ratio down
    feature['user_call_out_num_ratio'] = (feature['user_call_out_num'].fillna(0) + 0.001)/\
        (feature['user_call_num'].fillna(0) + 0.001)
    
    # user_call_day_num up
    t0 = voiceData.groupby('uid')['day'].nunique().reset_index().\
        rename(columns={'day':'user_call_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_max_num_one_day down
    t0 = voiceData[['uid','day']]
    t0['user_call_max_num_one_day'] = 1
    t0 = t0.groupby(['uid','day'])['user_call_max_num_one_day'].sum().reset_index()
    t0 = t0.groupby('uid')['user_call_max_num_one_day'].max().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_max_num_one_day
    t0 = voiceData[voiceData.call_in_out == 0][['uid','day']]
    t0['user_call_out_max_num_one_day'] = 1
    t0 = t0.groupby(['uid','day'])['user_call_out_max_num_one_day'].sum().reset_index()
    t0 = t0.groupby('uid')['user_call_out_max_num_one_day'].max().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_max_num_one_opp_num up
    t0 = voiceData[['uid','call_opp_num']]
    t0['user_call_max_num_one_opp_num'] = 1
    t0 = t0.groupby(['uid','call_opp_num'])['user_call_max_num_one_opp_num'].sum().reset_index()
    t0 = t0.groupby('uid')['user_call_max_num_one_opp_num'].max().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_max_num_one_opp_num_ratio
    feature['user_call_max_num_one_opp_num_ratio'] = (feature['user_call_max_num_one_opp_num'].fillna(0) + 0.001)/\
        (feature['user_call_num'].fillna(0) + 0.001)
    
    # user_call_out_max_num_one_opp_num up
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_opp_num']]
    t0['user_call_out_max_num_one_opp_num'] = 1
    t0 = t0.groupby(['uid','call_opp_num'])['user_call_out_max_num_one_opp_num'].sum().reset_index()
    t0 = t0.groupby('uid')['user_call_out_max_num_one_opp_num'].max().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_max_num_one_opp_num 
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_opp_num']]
    t0['user_call_in_max_num_one_opp_num'] = 1
    t0 = t0.groupby(['uid','call_opp_num'])['user_call_in_max_num_one_opp_num'].sum().reset_index()
    t0 = t0.groupby('uid')['user_call_in_max_num_one_opp_num'].max().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_different_opp_num
    t0 = voiceData.groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_different_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_different_opp_num
    t0 = voiceData[voiceData.call_in_out == 0].groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_out_different_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_different_opp_num down
    t0 = voiceData[voiceData.call_in_out == 1].groupby('uid')['call_opp_num'].nunique().reset_index().\
        rename(columns={'call_opp_num':'user_call_in_different_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_different_opp_num_ratio
    feature['user_call_out_different_opp_num_ratio'] = (feature['user_call_out_different_opp_num'].fillna(0) + 0.001)/\
        (feature['user_call_different_opp_num'].fillna(0) + 0.001)
        
    # user_call_opp_num_hot_value  up
    t0 = voiceData[['call_opp_num']]
    t0['call_opp_num_hot_value'] = 1
    t0 = t0.groupby('call_opp_num')['call_opp_num_hot_value'].sum().reset_index()
    t0['call_opp_num_hot_value'] = t0['call_opp_num_hot_value'].astype('float')/\
        t0['call_opp_num_hot_value'].sum()
    
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    
    t0 = voiceData.groupby('uid')['call_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_num_hot_value':'user_call_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_num_hot_value
    t0 = voiceData[voiceData.call_in_out == 0][['call_opp_num']]
    t0['call_out_opp_num_hot_value'] = 1
    t0 = t0.groupby('call_opp_num')['call_out_opp_num_hot_value'].sum().reset_index()
    t0['call_out_opp_num_hot_value'] = t0['call_out_opp_num_hot_value'].astype('float')/\
        t0['call_out_opp_num_hot_value'].sum()
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    
    t0 = voiceData.groupby('uid')['call_out_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_out_opp_num_hot_value':'user_call_out_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_opp_num_hot_value
    t0 = voiceData[voiceData.call_in_out == 1][['call_opp_num']]
    t0['call_in_opp_num_hot_value'] = 1
    t0 = t0.groupby('call_opp_num')['call_in_opp_num_hot_value'].sum().reset_index()
    t0['call_in_opp_num_hot_value'] = t0['call_in_opp_num_hot_value'].astype('float')/\
        t0['call_in_opp_num_hot_value'].sum()
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    
    t0 = voiceData.groupby('uid')['call_in_opp_num_hot_value'].sum().reset_index().\
        rename(columns={'call_in_opp_num_hot_value':'user_call_in_opp_num_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_max_day_num_one_opp_num
    t0 = voiceData.groupby('call_opp_num')['day'].nunique().reset_index().\
        rename(columns={'day':'call_opp_num_max_day_num'})
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    t0 = voiceData.groupby('uid')['call_opp_num_max_day_num'].max().reset_index().\
        rename(columns={'call_opp_num_max_day_num':'user_call_max_day_num_one_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_out_max_day_num_one_opp_num
    t0 = voiceData[voiceData.call_in_out == 0].groupby('call_opp_num')['day'].nunique().reset_index().\
        rename(columns={'day':'call_out_opp_num_max_day_num'})
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    t0 = voiceData.groupby('uid')['call_out_opp_num_max_day_num'].max().reset_index().\
        rename(columns={'call_out_opp_num_max_day_num':'user_call_out_max_day_num_one_opp_num'})
    feature = feature.merge(t0, on='uid', how='left')      
    
    # user_call_in_max_day_num_one_opp_num
    t0 = voiceData[voiceData.call_in_out == 1].groupby('call_opp_num')['day'].nunique().reset_index().\
        rename(columns={'day':'call_in_opp_num_max_day_num'})
    voiceData = voiceData.merge(t0, on='call_opp_num', how='left')
    t0 = voiceData.groupby('uid')['call_in_opp_num_max_day_num'].max().reset_index().\
        rename(columns={'call_in_opp_num_max_day_num':'user_call_in_max_day_num_one_opp_num'})
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_call_opp_num_cate_num
    t0 = voiceData.groupby('uid')['call_opp_head'].nunique().reset_index().\
        rename(columns={'call_opp_head':'user_call_opp_num_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_num_cate_num
    t0 = voiceData[voiceData.call_in_out == 0].groupby('uid')['call_opp_head'].nunique().reset_index().\
        rename(columns={'call_opp_head':'user_call_out_opp_num_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')   
    
    # user_call_in_opp_num_cate_num
    t0 = voiceData[voiceData.call_in_out == 1].groupby('uid')['call_opp_head'].nunique().reset_index().\
        rename(columns={'call_opp_head':'user_call_in_opp_num_cate_num'})
    feature = feature.merge(t0, on='uid', how='left') 

    # user_call_opp_head_hot_value
    t0 = voiceData[['call_opp_head']]
    t0['call_opp_head_hot_value'] = 1
    t0 = t0.groupby('call_opp_head')['call_opp_head_hot_value'].sum().reset_index()
    t0['call_opp_head_hot_value'] = t0['call_opp_head_hot_value'].astype('float')/\
        t0['call_opp_head_hot_value'].sum()
    voiceData = voiceData.merge(t0, on='call_opp_head', how='left')
    
    t0 = voiceData.groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_out_opp_head_hot_value
    t0 = voiceData[voiceData.call_in_out == 0].groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_out_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')

    # user_call_in_opp_head_hot_value
    t0 = voiceData[voiceData.call_in_out == 1].groupby('uid')['call_opp_head_hot_value'].sum().reset_index().\
        rename(columns={'call_opp_head_hot_value':'user_call_in_opp_head_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_in_opp_head_cate_1
    t0 = voiceData[['uid','call_opp_head']]
    t0['call_opp_head'] = t0['call_opp_head'].map(lambda s:1 if len(s) <2 else 0)
    t0 = t0.groupby('uid')['call_opp_head'].sum().reset_index().\
        rename(columns={'call_opp_head':'user_call_in_opp_head_cate_1'})
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
    
    # user_call_opp_len_12_ratio up
    feature['user_call_opp_len_12_ratio'] = (feature['user_call_opp_len_12'].fillna(0) + 0.001)/\
        (feature['user_call_num'].fillna(0) + 0.001)
    
    # user_call_out_opp_len_12
    t0 = voiceData[(voiceData.call_in_out == 0)&(voiceData.call_opp_len == 12)][['uid']]
    t0['user_call_out_opp_len_12'] = 1
    t0 = t0.groupby('uid')['user_call_out_opp_len_12'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_call_out_opp_len_12_ratio
    feature['user_call_out_opp_len_12_ratio'] = (feature['user_call_out_opp_len_12'].fillna(0) + 0.001)/\
        (feature['user_call_out_num'].fillna(0) + 0.001)
    
    # user_call_in_opp_len_12
    t0 = voiceData[(voiceData.call_in_out == 1)&(voiceData.call_opp_len == 12)][['uid']]
    t0['user_call_in_opp_len_12'] = 1
    t0 = t0.groupby('uid')['user_call_in_opp_len_12'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_call_opp_len_5
    t0 = voiceData[voiceData.call_opp_len == 5][['uid']]
    t0['user_call_opp_len_5'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_5'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_call_opp_len_8
    t0 = voiceData[voiceData.call_opp_len == 8][['uid']]
    t0['user_call_opp_len_8'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_8'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_opp_len_other
    t0 = voiceData[(voiceData.call_opp_len != 12)&(voiceData.call_opp_len != 11 ) \
                   &(voiceData.call_opp_len != 5)&(voiceData.call_opp_len != 8)][['uid']]
    t0['user_call_opp_len_other'] = 1
    t0 = t0.groupby('uid')['user_call_opp_len_other'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_total_time
    t0 = voiceData[['uid','call_start_time', 'call_end_time']]
    t0['call_start_time'] = t0['call_start_time']+':'+t0['call_end_time']
    t0['call_start_time'] = t0['call_start_time'].map(fun1)
    
    t0 = t0.groupby('uid')['call_start_time'].sum().reset_index().\
        rename(columns={'call_start_time':'user_call_total_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_mean_time
    feature['user_call_mean_time'] = (feature['user_call_total_time'].fillna(0) + 0.001)/\
        (feature['user_call_num'].fillna(0) + 0.001)
        
    # user_call_out_total_time
    t0 = voiceData[voiceData.call_in_out == 0][['uid','call_start_time', 'call_end_time']]
    t0['call_start_time'] = t0['call_start_time']+':'+t0['call_end_time']
    t0['call_start_time'] = t0['call_start_time'].map(fun1)
    
    t0 = t0.groupby('uid')['call_start_time'].sum().reset_index().\
        rename(columns={'call_start_time':'user_call_out_total_time'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_out_mean_time
    feature['user_call_out_mean_time'] = (feature['user_call_out_total_time'].fillna(0) + 0.001)/\
        (feature['user_call_out_num'].fillna(0) + 0.001)
        
    # user_call_in_total_time
    t0 = voiceData[voiceData.call_in_out == 1][['uid','call_start_time', 'call_end_time']]
    t0['call_start_time'] = t0['call_start_time']+':'+t0['call_end_time']
    t0['call_start_time'] = t0['call_start_time'].map(fun1)
    
    t0 = t0.groupby('uid')['call_start_time'].sum().reset_index().\
        rename(columns={'call_start_time':'user_call_in_total_time'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_call_in_mean_time
    feature['user_call_in_mean_time'] = (feature['user_call_in_total_time'].fillna(0) + 0.001)/\
        (feature['user_call_in_num'].fillna(0) + 0.001)
    
    # user_call_max_time
    t0 = voiceData[['uid','call_start_time', 'call_end_time']]
    t0['call_start_time'] = t0['call_start_time']+':'+t0['call_end_time']
    t0['call_start_time'] = t0['call_start_time'].map(fun1)
    
    t0 = t0.groupby('uid')['call_start_time'].max().reset_index().\
        rename(columns={'call_start_time':'user_call_max_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_median_time
    t0 = voiceData[['uid','call_start_time', 'call_end_time']]
    t0['call_start_time'] = t0['call_start_time']+':'+t0['call_end_time']
    t0['call_start_time'] = t0['call_start_time'].map(fun1)
    
    t0 = t0.groupby('uid')['call_start_time'].median().reset_index().\
        rename(columns={'call_start_time':'user_call_median_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_type_1
    t0 = voiceData[voiceData.call_type == 1][['uid']]
    t0['user_call_type_1'] = 1
    t0 = t0.groupby('uid')['user_call_type_1'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_type_2
    t0 = voiceData[voiceData.call_type == 2][['uid']]
    t0['user_call_type_2'] = 1
    t0 = t0.groupby('uid')['user_call_type_2'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')
        
    # user_call_type_3
    t0 = voiceData[voiceData.call_type.isin([3,4,5])][['uid']]
    t0['user_call_type_3'] = 1
    t0 = t0.groupby('uid')['user_call_type_3'].sum().reset_index()
    feature = feature.merge(t0, on='uid', how='left')  
           
    # 45天内通话间隔统计特征
    t = voiceData.groupby('uid')['call_start_time_sec'].agg(lambda x: ':'.join(x)).reset_index()
    
    t['user_call_during_min'] = t['call_start_time_sec'].map(lambda s: fun5(s, 0))
    t['user_call_during_max'] = t['call_start_time_sec'].map(lambda s: fun5(s, 1))
    t['user_call_during_mean'] = t['call_start_time_sec'].map(lambda s: fun5(s, 2))    
    t['user_call_during_median'] = t['call_start_time_sec'].map(lambda s: fun5(s, 3))
    t['user_call_during_std'] = t['call_start_time_sec'].map(lambda s: fun5(s, 4))
    t['user_call_during_zero'] = t['call_start_time_sec'].map(lambda s: fun5(s, 6))
    
    t.drop(['call_start_time_sec'], axis=1, inplace=True)

    feature = feature.merge(t, on='uid', how='left')
    
    return feature

def get_user_call_num_at_day(voiceData, in_out):
    if in_out not in [0,1]:
        t = voiceData[['uid','day']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','day']]
        
    for day in range(1, 46):
        s = 'user_call_%s_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.day == day][['uid']]
        t0[s] = 1
        t0 = t0.groupby('uid')[s].sum().reset_index()
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['day'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t

def get_user_call_num_at_week(voiceData, in_out):
    if in_out not in [0,1]:
        t = voiceData[['uid','weeknum']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','weeknum']]
    
    t['user_call_num_at_weeknum'] = 1
    t = t.groupby(['uid','weeknum'])['user_call_num_at_weeknum'].sum().reset_index()
    
    for week in range(1,8):
        s = 'user_call_%s_num_at_week_%s' % (str(in_out), str(week))
        t0 = t[t.weeknum == week][['uid','user_call_num_at_weeknum']]
        t0.rename(columns={'user_call_num_at_weeknum':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['weeknum','user_call_num_at_weeknum'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('median', np.median), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_num_at_week_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t

def get_user_call_num_at_hour(voiceData, in_out):
    if in_out not in [0,1]:
        t = voiceData[['uid','call_start_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','call_start_time']]
    
    t['call_start_time'] = t['call_start_time'].map(lambda s: int(s[2:4]))
        
    for hour in range(0, 24):
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
    
    return t

def get_user_call_different_people_num_at_day(voiceData, in_out):
    if in_out not in [0,1]:
        t = voiceData[['uid','day','call_opp_num']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','day','call_opp_num']]
    
    t = t.groupby(['uid','day'])['call_opp_num'].nunique().reset_index()
    
    for day in range(1, 46):
        s = 'user_call_%s_different_people_num_at_day_%s' % (str(in_out), str(day))
        t0 = t[t.day == day][['uid','call_opp_num']]
        t0.rename(columns={'call_opp_num':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
        
    t.drop(['day','call_opp_num'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_different_people_num_at_day_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val    
    
    return t

def get_user_call_num_at_weekday(voiceData, in_out):
    if in_out not in [0,1]:
        t = voiceData[['uid','weekday','call_time']]
    else:
        t = voiceData[voiceData.call_in_out == in_out][['uid','weekday','call_time']]
    
    t = t.groupby(['uid','weekday'])['call_time'].count().reset_index()
        
    for day in range(1, 8):
        s = 'user_call_%s_num_at_weekday_%s' % (str(in_out), str(day))
        t0 = t[t.weekday == day][['uid', 'call_time']]
        t0.rename(columns={'call_time':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['weekday','call_time'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_call_%s_num_at_weekday_%s' % (str(in_out), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t    
    
    

def fun2(s):
    s = [int(n) for n in s.split(':')]
    s = sorted(list(set(s)))# 去重并排序
    if len(s) == 1:
        return max(45-s[0], s[0]-1)
    else:
        res = []
        for i in range(1, len(s)):
            res.append(s[i]-s[i-1])
        return max(res)

def fun3(s):
    s = [int(n) for n in s.split(':')]
    s = sorted(list(set(s)))# 去重并排序
    if len(s) == 1:
        return max(45-s[0], s[0]-1)
    else:
        res = []
        for i in range(1, len(s)):
            res.append(s[i]-s[i-1])
        return np.mean(res)    

def fun4(s):    
    return int(s[0:2])*24*60*60 + int(s[2:4])*60*60 + int(s[4:6])*60 + int(s[6:8])

voiceData['call_start_time_sec'] = voiceData['call_start_time'].map(fun4).astype('str')

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

def make_voice_feature2(voiceData, feature):
    # user_call_x_num_at_y
    t0 = get_user_call_num_at_day(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_num_at_week_y
    t0 = get_user_call_num_at_week(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # 最大间隔
    t0 = voiceData[['uid','day']]
    t0['day'] = t0['day'].astype('str')
    t0 = t0.groupby('uid')['day'].agg(lambda x: ':'.join(x)).reset_index()
    t0['day'] = t0['day'].map(fun2)
    t0.rename(columns={'day':'user_call_max_time_delta'}, inplace=True)
    feature = feature.merge(t0, on='uid', how='left')
    
    # 平均间隔
    t0 = voiceData[['uid','day']]
    t0['day'] = t0['day'].astype('str')
    t0 = t0.groupby('uid')['day'].agg(lambda x: ':'.join(x)).reset_index()
    t0['day'] = t0['day'].map(fun3)
    t0.rename(columns={'day':'user_call_mean_time_delta'}, inplace=True)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_num_at_hour_y
    t0 = get_user_call_num_at_hour(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_different_people_num_at_day_y
    t0 = get_user_call_different_people_num_at_day(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_call_x_num_at_weekday_y
    t0 = get_user_call_num_at_weekday(voiceData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    

    
    return feature

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('../data/uid_train.txt', header = None)
uidTrain.columns = ['uid', 'label']

uidTest = pd.DataFrame()
uidTest['uid'] = range(5000,10000)
uidTest.uid = uidTest.uid.apply(lambda x: 'u'+str(x).zfill(4))

feature = pd.concat([uidTrain.drop('label', axis=1), uidTest])

# 提取特征
feature = make_voice_feature(voiceData, feature)
#feature = pd.read_csv('../data/feature_voice_Btest.csv', header = 0)

#feature = make_voice_feature2(voiceData, feature)
# feature = pd.read_csv('../data/feature_voice_Btest2.csv', header = 0)

# feature.to_csv('../data/feature_voice_Btest03.csv', index=False)

# 训练集
train = feature[:4999].copy()
train = train.merge(uidTrain, on='uid', how='left')

# 打乱顺序
np.random.seed(201805)
idx = np.random.permutation(len(train))
train = train.iloc[idx]

X_train = train.drop(['uid','label'], axis=1).values
y_train = train.label.values

lgb = lightgbm.LGBMClassifier(random_state=201806)

lr = LogisticRegression(C=10, random_state=201806)

def fitModel(model, feature1):
    X = feature1.drop(['uid','label'], axis=1).values
    y = feature1.label.values
    
    #X2 = feature1.fillna(0).drop(['uid','label'], axis=1).values
    
    #lgb_y_pred = cross_val_predict(model, X, y, cv=5,
    #                       verbose=2, method='predict')
    lgb_y_proba = cross_val_predict(model, X, y, cv=5,
                                verbose=2, method='predict_proba')[:,1]
    
    #lr_y_pred = cross_val_predict(lr, X2, y, cv=5,
    #                       verbose=2, method='predict')
    #lr_y_proba = cross_val_predict(lr, X2, y, cv=5,
    #                           verbose=2, method='predict_proba')[:,1]
    # 融合lr
    #lgb_y_proba = lgb_y_proba * 0.9 + lr_y_proba * 0.1
    
    lgb_y_pred  =  (lgb_y_proba >= 0.5)*1
    
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
xgb = xgboost.XGBClassifier(random_state=201806)
#fitModel(xgb, train)


"""
F1: 0.129400570885 AUC: 0.673206337589 Score: 0.455684030907
             precision    recall  f1-score   support

          0       0.83      0.98      0.90      4099
          1       0.45      0.08      0.13       900

avg / total       0.76      0.82      0.76      4999

             feature  score
0  user_call_out_num   1079
1   user_call_in_num    982
2      user_call_num    939

F1: 0.216494845361 AUC: 0.716614485918 Score: 0.516566629695
             precision    recall  f1-score   support

          0       0.84      0.97      0.90      4099
          1       0.48      0.14      0.22       900

avg / total       0.77      0.82      0.77      4999

                   feature  score
0            user_call_num    677
1  user_call_out_num_ratio    656
2         user_call_in_num    635
3        user_call_out_num    534
4        user_call_day_num    498


F1: 0.23654015887 AUC: 0.723122035185 Score: 0.528489284659
             precision    recall  f1-score   support

          0       0.84      0.98      0.90      4099
          1       0.58      0.15      0.24       900

avg / total       0.79      0.83      0.78      4999

                         feature  score
0  user_call_max_num_one_opp_num    571
1        user_call_out_num_ratio    439
2              user_call_out_num    424
3               user_call_in_num    423
4                  user_call_num    404
5      user_call_max_num_one_day    385
6              user_call_day_num    354

F1: 0.251089799477 AUC: 0.73416537909 Score: 0.540935147245
             precision    recall  f1-score   support

          0       0.84      0.97      0.90      4099
          1       0.58      0.16      0.25       900

avg / total       0.79      0.83      0.79      4999

                             feature  score
0                   user_call_in_num    353
1   user_call_in_max_num_one_opp_num    328
2                      user_call_num    320
3  user_call_out_max_num_one_opp_num    320
4                  user_call_out_num    315
5            user_call_out_num_ratio    312
6      user_call_max_num_one_opp_num    306
7                  user_call_day_num    298
8          user_call_max_num_one_day    236
9      user_call_out_max_num_one_day    212

F1: 0.254134029591 AUC: 0.74532501152 Score: 0.548848618749
             precision    recall  f1-score   support

          0       0.84      0.97      0.90      4099
          1       0.59      0.16      0.25       900

avg / total       0.80      0.83      0.79      4999

                              feature  score
0         user_call_different_opp_num    455
1             user_call_out_num_ratio    319
2   user_call_out_max_num_one_opp_num    295
3                   user_call_day_num    292
4       user_call_max_num_one_opp_num    289
5                    user_call_in_num    288
6    user_call_in_max_num_one_opp_num    275
7                       user_call_num    262
8                   user_call_out_num    212
9           user_call_max_num_one_day    166
10      user_call_out_max_num_one_day    147

F1: 0.275213675214 AUC: 0.743722588165 Score: 0.556319022985
             precision    recall  f1-score   support

          0       0.84      0.97      0.90      4099
          1       0.60      0.18      0.28       900

avg / total       0.80      0.83      0.79      4999

                              feature  score
0         user_call_different_opp_num    363
1       user_call_max_num_one_opp_num    289
2                    user_call_in_num    284
3                   user_call_day_num    275
4             user_call_out_num_ratio    264
5   user_call_out_max_num_one_opp_num    261
6                       user_call_num    247
7    user_call_in_max_num_one_opp_num    243
8     user_call_out_different_opp_num    243
9                   user_call_out_num    193
10          user_call_max_num_one_day    190
11      user_call_out_max_num_one_day    148


F1: 0.336283185841 AUC: 0.767397603752 Score: 0.594951836587
             precision    recall  f1-score   support

          0       0.85      0.97      0.91      4099
          1       0.61      0.23      0.34       900

avg / total       0.81      0.83      0.80      4999

                                  feature  score
0             user_call_opp_num_hot_value    463
1   user_call_out_different_opp_num_ratio    259
2                       user_call_day_num    231
3                        user_call_in_num    215
4                 user_call_out_num_ratio    206
5        user_call_in_max_num_one_opp_num    196
6             user_call_different_opp_num    194
7       user_call_out_max_num_one_opp_num    185
8                           user_call_num    179
9           user_call_max_num_one_opp_num    176
10         user_call_in_different_opp_num    169
11        user_call_out_different_opp_num    154
12                      user_call_out_num    145
13              user_call_max_num_one_day    126
14          user_call_out_max_num_one_day    102

F1: 0.349639133921 AUC: 0.77037353284 Score: 0.602079773273
             precision    recall  f1-score   support

          0       0.85      0.97      0.91      4099
          1       0.63      0.24      0.35       900

avg / total       0.81      0.84      0.81      4999

                                  feature  score
0          user_call_in_opp_num_hot_value    291
1         user_call_out_opp_num_hot_value    230
2             user_call_opp_num_hot_value    225
3                       user_call_day_num    206
4   user_call_out_different_opp_num_ratio    188
5           user_call_max_num_one_opp_num    184
6             user_call_different_opp_num    181
7                        user_call_in_num    179
8                 user_call_out_num_ratio    174
9          user_call_in_different_opp_num    166
10                          user_call_num    165
11       user_call_in_max_num_one_opp_num    163
12      user_call_out_max_num_one_opp_num    156
13        user_call_out_different_opp_num    153
14                      user_call_out_num    146
15              user_call_max_num_one_day    108
16          user_call_out_max_num_one_day     85

F1: 0.368839427663 AUC: 0.773534466401 Score: 0.611656450906
             precision    recall  f1-score   support

          0       0.86      0.97      0.91      4099
          1       0.65      0.26      0.37       900

avg / total       0.82      0.84      0.81      4999

                                  feature  score
0          user_call_in_opp_num_hot_value    263
1                       user_call_day_num    221
2   user_call_out_different_opp_num_ratio    189
3             user_call_opp_num_hot_value    187
4                        user_call_in_num    179
5        user_call_in_max_num_one_opp_num    177
6           user_call_max_num_one_opp_num    170
7         user_call_out_opp_num_hot_value    169
8                 user_call_out_num_ratio    164
9             user_call_different_opp_num    157
10      user_call_out_max_num_one_opp_num    157
11                          user_call_num    148
12         user_call_in_different_opp_num    146
13        user_call_out_different_opp_num    141
14              user_call_max_num_one_day    121
15                      user_call_out_num    116
16          user_call_out_max_num_one_day     79
17      user_call_max_day_num_one_opp_num     73
18  user_call_out_max_day_num_one_opp_num     72
19   user_call_in_max_day_num_one_opp_num     71


F1: 0.385376999238 AUC: 0.821325526551 Score: 0.646946115626
             precision    recall  f1-score   support

          0       0.86      0.96      0.91      4099
          1       0.61      0.28      0.39       900

avg / total       0.81      0.84      0.81      4999

                                  feature  score
0            user_call_opp_head_hot_value    180
1         user_call_in_opp_head_hot_value    179
2   user_call_out_different_opp_num_ratio    163
3     user_call_max_num_one_opp_num_ratio    161
4        user_call_out_opp_head_hot_value    147
5                    user_call_opp_len_12    144
6          user_call_in_opp_num_hot_value    144
7         user_call_out_opp_num_hot_value    128
8                       user_call_day_num    120
9                 user_call_out_num_ratio    118
10            user_call_different_opp_num    111
11            user_call_opp_num_hot_value    110
12             user_call_opp_num_cate_num    105
13         user_call_in_different_opp_num     97
14       user_call_in_max_num_one_opp_num     96
15          user_call_in_opp_num_cate_num     95
16      user_call_out_max_num_one_opp_num     83
17                       user_call_in_num     80
18                   user_call_opp_len_11     76
19         user_call_out_opp_num_cate_num     75
20              user_call_max_num_one_day     68
21        user_call_out_different_opp_num     67
22          user_call_max_num_one_opp_num     65
23                          user_call_num     59
24          user_call_out_max_num_one_day     57
25      user_call_max_day_num_one_opp_num     57
26                      user_call_out_num     57
27           user_call_in_opp_head_cate_1     56
28   user_call_in_max_day_num_one_opp_num     51
29  user_call_out_max_day_num_one_opp_num     51

F1: 0.406906906907 AUC: 0.824272857879 Score: 0.65732647749
             precision    recall  f1-score   support

          0       0.86      0.96      0.91      4099
          1       0.63      0.30      0.41       900

avg / total       0.82      0.84      0.82      4999

                                  feature  score
0                     user_call_mean_time    180
1         user_call_in_opp_head_hot_value    163
2   user_call_out_different_opp_num_ratio    138
3         user_call_out_opp_num_hot_value    130
4                    user_call_total_time    124
5                       user_call_day_num    120
6     user_call_max_num_one_opp_num_ratio    118
7            user_call_opp_head_hot_value    113
8          user_call_in_opp_num_hot_value    109
9              user_call_opp_len_12_ratio     97
10            user_call_opp_num_hot_value     91
11                       user_call_in_num     90
12         user_call_in_different_opp_num     87
13       user_call_in_max_num_one_opp_num     77
14            user_call_different_opp_num     77
15       user_call_out_opp_head_hot_value     77
16                    user_call_opp_len_8     76
17             user_call_opp_num_cate_num     73
18                user_call_opp_len_other     73
19                user_call_out_num_ratio     73
20      user_call_out_max_num_one_opp_num     70
21          user_call_max_num_one_opp_num     69
22          user_call_in_opp_num_cate_num     64
23                   user_call_opp_len_11     64
24                    user_call_opp_len_5     59
25                user_call_in_opp_len_12     57
26      user_call_max_day_num_one_opp_num     56
27   user_call_in_max_day_num_one_opp_num     49
28              user_call_max_num_one_day     48
29         user_call_out_opp_len_12_ratio     47
30         user_call_out_opp_num_cate_num     46
31          user_call_out_max_num_one_day     44
32                      user_call_out_num     41
33        user_call_out_different_opp_num     41
34  user_call_out_max_day_num_one_opp_num     40
35                          user_call_num     39
36                   user_call_opp_len_12     37
37           user_call_in_opp_head_cate_1     31
38               user_call_out_opp_len_12     12

F1: 0.402714932127 AUC: 0.829120652734 Score: 0.658558364491
             precision    recall  f1-score   support

          0       0.86      0.96      0.91      4099
          1       0.63      0.30      0.40       900

avg / total       0.82      0.84      0.82      4999

                                  feature  score
0                      user_call_max_time    160
1                   user_call_median_time    125
2                        user_call_type_3    116
3         user_call_in_opp_head_hot_value    111
4         user_call_out_opp_num_hot_value    105
5   user_call_out_different_opp_num_ratio    101
6                 user_call_in_total_time    100
7                       user_call_day_num     92
8          user_call_in_opp_num_hot_value     91
9            user_call_opp_head_hot_value     91
10                    user_call_mean_time     90
11    user_call_max_num_one_opp_num_ratio     87
12                       user_call_type_2     83
13                user_call_out_mean_time     81
14             user_call_opp_len_12_ratio     76
15               user_call_out_total_time     76
16                       user_call_type_1     70
17       user_call_out_opp_head_hot_value     70
18                user_call_out_num_ratio     69
19            user_call_different_opp_num     69
20                 user_call_in_mean_time     66
21             user_call_opp_num_cate_num     62
22                user_call_opp_len_other     61
23         user_call_in_different_opp_num     61
24            user_call_opp_num_hot_value     60
25                    user_call_opp_len_8     59
26                   user_call_total_time     52
27      user_call_out_max_num_one_opp_num     51
28          user_call_in_opp_num_cate_num     50
29        user_call_out_different_opp_num     44
30       user_call_in_max_num_one_opp_num     43
31                   user_call_opp_len_11     41
32                user_call_in_opp_len_12     41
33                       user_call_in_num     40
34  user_call_out_max_day_num_one_opp_num     37
35                   user_call_opp_len_12     36
36         user_call_out_opp_len_12_ratio     36
37      user_call_max_day_num_one_opp_num     36
38              user_call_max_num_one_day     35
39         user_call_out_opp_num_cate_num     34
40   user_call_in_max_day_num_one_opp_num     32
41                    user_call_opp_len_5     31
42          user_call_out_max_num_one_day     28
43          user_call_max_num_one_opp_num     28
44                          user_call_num     24
45           user_call_in_opp_head_cate_1     23
46                      user_call_out_num     22
47               user_call_out_opp_len_12      4
"""










