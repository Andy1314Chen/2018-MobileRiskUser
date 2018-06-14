# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:19:57 2018

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

# 假设第一天是周5
weekList = [5,6,7,1,2,3,4]*7
weekList = weekList[:45]

waData = pd.concat([waTrain, waTest])
#waData['weekday'] = waData['date'].astype('int').map(lambda x: weekList[x-1] if x >0 else 1)

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('../data/uid_train.txt', header = None)
uidTrain.columns = ['uid', 'label']

uidTest = pd.DataFrame()
uidTest['uid'] = range(5000,7000)
uidTest.uid = uidTest.uid.apply(lambda x: 'u'+str(x).zfill(4))

feature = pd.concat([uidTrain.drop('label', axis=1), uidTest])

def make_user_wa_feature(waData, feature):
    # user_visit_web_cate_num
    t0 = waData[waData.wa_type == 0][['uid','wa_name']]
    t0 = t0.groupby('uid')['wa_name'].nunique().reset_index()
    
    feature = feature.merge(t0, on='uid', how='left')
    
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














