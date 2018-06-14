# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:31:24 2018

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


# wa_train.txt 用户网站访问记录数据
waTrain = pd.read_table('../data/wa_train.txt', header = None)
waTrain.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow',
                   'down_flow', 'wa_type', 'date']
waTest = pd.read_table('../data/wa_test_a.txt', header = None)
waTest.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow',
                   'down_flow', 'wa_type', 'date']

waTestb = pd.read_table('../data/wa_test_b.txt', header = None)
waTestb.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow',
                   'down_flow', 'wa_type', 'date']

# 假设第一天是周5
weekList = [5,6,7,1,2,3,4]*7
weekList = weekList[:45]

waData = pd.concat([waTrain, waTest])
waData = pd.concat([waData, waTestb])
#waData['weekday'] = waData['date'].astype('int').map(lambda x: weekList[x-1] if x >0 else 1)

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('../data/uid_train.txt', header = None)
uidTrain.columns = ['uid', 'label']

uidTest = pd.DataFrame()
uidTest['uid'] = range(5000,7000)
uidTest.uid = uidTest.uid.apply(lambda x: 'u'+str(x).zfill(4))

feature = pd.concat([uidTrain.drop('label', axis=1), uidTest])


def get_user_visit_num_at_day(waData, wa_type):
    if wa_type not in [0,1]:
        t = waData[['uid','date']]
    else:
        t = waData[waData.wa_type == wa_type][['uid','date']]
        
    t['visit_num']  = 1
    t = t.groupby(['uid','date'])['visit_num'].sum().reset_index()
    
    for day in range(1, 46):
        s = 'user_visit_%s_num_at_day_%s' % (str(wa_type), str(day))
        t0 = t[t.date == day][['uid','visit_num']]
        t0.rename(columns={'visit_num':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['date','visit_num'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_visit_%s_num_at_day_%s' % (str(wa_type), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t    

def get_user_visit_time_at_day(waData, wa_type):
    if wa_type not in [0,1]:
        t = waData[['uid','date','visit_dura']]
    else:
        t = waData[waData.wa_type == wa_type][['uid','date','visit_dura']]
        
    t = t.groupby(['uid','date'])['visit_dura'].sum().reset_index()
    
    for day in range(1, 46):
        s = 'user_visit_%s_time_at_day_%s' % (str(wa_type), str(day))
        t0 = t[t.date == day][['uid','visit_dura']]
        t0.rename(columns={'visit_dura':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['date', 'visit_dura'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_visit_%s_time_at_day_%s' % (str(wa_type), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t      

def get_user_visit_flow_at_day(waData, wa_type):
    if wa_type not in [0,1]:
        t = waData[['uid','date','down_flow']]
    else:
        t = waData[waData.wa_type == wa_type][['uid','date','down_flow']]
        
    t = t.groupby(['uid','date'])['down_flow'].sum().reset_index()
    
    for day in range(1, 46):
        s = 'user_visit_%s_flow_at_day_%s' % (str(wa_type), str(day))
        t0 = t[t.date == day][['uid','down_flow']]
        t0.rename(columns={'down_flow':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['date', 'down_flow'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_visit_%s_flow_at_day_%s' % (str(wa_type), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val
    
    return t 

def get_user_visit_up_flow_at_day(waData, wa_type):
    if wa_type not in [0,1]:
        t = waData[['uid','date','up_flow']]
    else:
        t = waData[waData.wa_type == wa_type][['uid','date','up_flow']]
        
    t = t.groupby(['uid','date'])['up_flow'].sum().reset_index()
    
    for day in range(1, 46):
        s = 'user_visit_%s_up_flow_at_day_%s' % (str(wa_type), str(day))
        t0 = t[t.date == day][['uid','up_flow']]
        t0.rename(columns={'up_flow':s}, inplace=True)
        t = t.merge(t0, on='uid', how='left')
    
    t.drop(['date', 'up_flow'], axis=1, inplace=True)
    t.fillna(0, inplace=True)
    t.drop_duplicates(inplace=True)
    
    df = t.drop('uid', axis=1)
    
    for fun in [('mean', np.mean), ('max',np.max), ('std', np.std), ('var', np.var)]:
        s = 'user_visit_%s_up_flow_at_day_%s' % (str(wa_type), fun[0])
        val = df.apply(fun[1], axis=1).values
        t[s] = val    
    
    return t


def make_user_wa_feature(waData, feature):
    # user_visit_web_cate_num
    t0 = waData[waData.wa_type == 0][['uid','wa_name']]
    t0 = t0.groupby('uid')['wa_name'].nunique().reset_index().\
        rename(columns={'wa_name':'user_visit_web_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_app_cate_num
    t0 = waData[waData.wa_type == 1][['uid','wa_name']]
    t0 = t0.groupby('uid')['wa_name'].nunique().reset_index().\
        rename(columns={'wa_name':'user_visit_app_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')  
    
    # user_visit_wa_cate_num
    t0 = waData[['uid','wa_name']]
    t0 = t0.groupby('uid')['wa_name'].nunique().reset_index().\
        rename(columns={'wa_name':'user_visit_wa_cate_num'})
    feature = feature.merge(t0, on='uid', how='left')      
    
    # user_visit_num
    t0 = waData.groupby('uid')['date'].count().reset_index().\
        rename(columns={'date':'user_visit_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_app_num
    t0 = waData[waData.wa_type == 1].groupby('uid')['date'].count().reset_index().\
        rename(columns={'date':'user_visit_app_num'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_visit_web_num
    t0 = waData[waData.wa_type == 0].groupby('uid')['date'].count().reset_index().\
        rename(columns={'date':'user_visit_web_num'})
    feature = feature.merge(t0, on='uid', how='left') 

    # user_visit_web_num_ratio
    feature['user_visit_web_num_ratio'] = (feature['user_visit_web_num'].fillna(0) + 0.001)/\
    (feature['user_visit_num'].fillna(0) + 0.001)    
    
    # user_visit_app_hot_value
    t0 = waData[waData.wa_type == 1][['wa_name']]
    t0['app_hot_value'] = 1
    t0 = t0.groupby('wa_name')['app_hot_value'].sum().reset_index()
    t0['app_hot_value'] = t0['app_hot_value'].astype('float')/\
        t0['app_hot_value'].sum()
    waData = waData.merge(t0, on='wa_name', how='left')
    
    t0 = waData.groupby('uid')['app_hot_value'].sum().reset_index().\
        rename(columns={'app_hot_value':'user_visit_app_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_web_hot_value
    t0 = waData[waData.wa_type == 0][['wa_name']]
    t0['web_hot_value'] = 1
    t0 = t0.groupby('wa_name')['web_hot_value'].sum().reset_index()
    t0['web_hot_value'] = t0['web_hot_value'].astype('float')/\
        t0['web_hot_value'].sum()
    waData = waData.merge(t0, on='wa_name', how='left')
    
    t0 = waData.groupby('uid')['web_hot_value'].sum().reset_index().\
        rename(columns={'web_hot_value':'user_visit_web_hot_value'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_app_day_num
    t0 = waData[waData.wa_type == 1][['uid','date']]
    t0 = t0.groupby('uid')['date'].nunique().reset_index().\
        rename(columns={'date':'user_visit_app_day_num'})
    feature = feature.merge(t0, on='uid', how='left')   
    
    # user_visit_web_day_num
    t0 = waData[waData.wa_type == 0][['uid','date']]
    t0 = t0.groupby('uid')['date'].nunique().reset_index().\
        rename(columns={'date':'user_visit_web_day_num'})
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_visit_day_num
    t0 = waData.groupby('uid')['date'].nunique().reset_index().\
        rename(columns={'date':'user_visit_day_num'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_app_total_cnt
    t0 = waData[waData.wa_type == 1].groupby('uid')['visit_cnt'].sum().reset_index().\
        rename(columns={'visit_cnt':'user_visit_app_total_cnt'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_web_total_cnt
    t0 = waData[waData.wa_type == 0].groupby('uid')['visit_cnt'].sum().reset_index().\
        rename(columns={'visit_cnt':'user_visit_web_total_cnt'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_visit_app_total_time
    t0 = waData[waData.wa_type == 1].groupby('uid')['visit_dura'].sum().reset_index().\
        rename(columns={'visit_dura':'user_visit_app_total_time'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_web_total_time
    t0 = waData[waData.wa_type == 0].groupby('uid')['visit_dura'].sum().reset_index().\
        rename(columns={'visit_dura':'user_visit_web_total_time'})
    feature = feature.merge(t0, on='uid', how='left')   
    
    # user_visit_app_mean_time
    feature['user_visit_app_mean_time'] = (feature['user_visit_app_total_time'].fillna(0) + 0.001)/\
        (feature['user_visit_app_total_cnt'].fillna(0) + 0.001)
        
    # user_visit_web_mean_time
    feature['user_visit_web_mean_time'] = (feature['user_visit_web_total_time'].fillna(0) + 0.001)/\
        (feature['user_visit_web_total_cnt'].fillna(0) + 0.001)  
        
    # user_visit_app_down_flow
    t0 = waData[waData.wa_type == 1].groupby('uid')['down_flow'].sum().reset_index().\
        rename(columns={'down_flow':'user_visit_app_down_flow'})
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_web_down_flow
    t0 = waData[waData.wa_type == 0].groupby('uid')['down_flow'].sum().reset_index().\
        rename(columns={'down_flow':'user_visit_web_down_flow'})
    feature = feature.merge(t0, on='uid', how='left')

    # user_visit_app_up_flow
    t0 = waData[waData.wa_type == 1].groupby('uid')['up_flow'].sum().reset_index().\
        rename(columns={'up_flow':'user_visit_app_up_flow'})
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_visit_web_up_flow
    t0 = waData[waData.wa_type == 0].groupby('uid')['up_flow'].sum().reset_index().\
        rename(columns={'up_flow':'user_visit_web_up_flow'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_visit_app_max_cnt
    t0 = waData[waData.wa_type == 1].groupby(['uid','wa_name'])['visit_cnt'].sum().reset_index()
    t0 = t0.groupby('uid')['visit_cnt'].max().reset_index().\
        rename(columns={'visit_cnt':'user_visit_app_max_cnt'})
    feature = feature.merge(t0, on='uid', how='left')
        
    # user_visit_web_max_cnt
    t0 = waData[waData.wa_type == 0].groupby(['uid','wa_name'])['visit_cnt'].sum().reset_index()
    t0 = t0.groupby('uid')['visit_cnt'].max().reset_index().\
        rename(columns={'visit_cnt':'user_visit_web_max_cnt'}) 
    feature = feature.merge(t0, on='uid', how='left')
        
    # user_visit_app_max_time
    t0 = waData[waData.wa_type == 1].groupby(['uid','wa_name'])['visit_dura'].sum().reset_index()
    t0 = t0.groupby('uid')['visit_dura'].max().reset_index().\
        rename(columns={'visit_dura':'user_visit_app_max_time'})
    feature = feature.merge(t0, on='uid', how='left')    
    
    # user_visit_web_max_time
    t0 = waData[waData.wa_type == 0].groupby(['uid','wa_name'])['visit_dura'].sum().reset_index()
    t0 = t0.groupby('uid')['visit_dura'].max().reset_index().\
        rename(columns={'visit_dura':'user_visit_web_max_time'})
    feature = feature.merge(t0, on='uid', how='left')

    # user_visit_app_max_down_flow   
    t0 = waData[waData.wa_type == 1].groupby(['uid','wa_name'])['down_flow'].sum().reset_index()
    t0 = t0.groupby('uid')['down_flow'].max().reset_index().\
        rename(columns={'down_flow':'user_visit_app_max_down_flow'})
    feature = feature.merge(t0, on='uid', how='left') 
    
    # user_visit_web_max_down_flow
    t0 = waData[waData.wa_type == 0].groupby(['uid','wa_name'])['down_flow'].sum().reset_index()
    t0 = t0.groupby('uid')['down_flow'].max().reset_index().\
        rename(columns={'down_flow':'user_visit_web_max_down_flow'})
    feature = feature.merge(t0, on='uid', how='left')     
    
    # user_visit_0_num_at_day_y
    t0 = get_user_visit_num_at_day(waData, 0)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_1_num_at_day_y
    t0 = get_user_visit_num_at_day(waData, 1)
    feature = feature.merge(t0, on='uid', how='left')
      
    # user_visit_0_time_at_day_y
    t0 = get_user_visit_time_at_day(waData, 0)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_1_time_at_day_y
    t0 = get_user_visit_time_at_day(waData, 1)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_0_flow_at_day_y
    t0 = get_user_visit_flow_at_day(waData, 0)
    feature = feature.merge(t0, on='uid', how='left')
    
    # user_visit_1_flow_at_day_y
    t0 = get_user_visit_flow_at_day(waData, 1)
    feature = feature.merge(t0, on='uid', how='left')   
    
    t0 = get_user_visit_up_flow_at_day(waData, 2)
    feature = feature.merge(t0, on='uid', how='left')
    
    # 每天同一时刻发出的短信数量 = 每天总量 - 每天发生时间数
        
    return feature

# 提取特征
feature = make_user_wa_feature(waData, feature)

# feature.to_csv('../data/feature_wa_03.csv', index=False)

# 训练集
train = feature[:4999].copy()
train = train.merge(uidTrain, on='uid', how='left')

# 打乱顺序
np.random.seed(201805)
idx = np.random.permutation(len(train))
train = train.iloc[idx]

X_train = train.drop(['uid','label'], axis=1).values
y_train = train.label.values


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
    featureImportant['score'] = model.feature_importances_
    featureImportant.sort_values(by='score', ascending=False, inplace=True)
    featureImportant.reset_index(drop=True, inplace=True)
    print(featureImportant)
    


# 交叉验证模型
fitModel(lgb, train)