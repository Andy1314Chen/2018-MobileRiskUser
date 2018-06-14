# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:02:48 2018

@author: MSIK
"""

import pandas as pd
import numpy as np
from datetime import date

import warnings
warnings.filterwarnings('ignore')

import lightgbm
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_predict

from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 提取的400+维特征
feature_01 = pd.read_csv('../data/feature_voice_03b.csv', header = 0)#Btest03
feature_02 = pd.read_csv('../data/feature_sms_04b.csv', header = 0)#03b #testB03
feature_03 = pd.read_csv('../data/feature_wa_03b.csv', header = 0)#testB03

feature = feature_01.merge(feature_02, on='uid', how='left')
feature = feature.merge(feature_03, on='uid', how='left')
feature.fillna(0, inplace=True)

# 标签uid_train.txt  0:4099 1:900
uidTrain = pd.read_table('../data/uid_train.txt', header = None)
uidTrain.columns = ['uid', 'label']

# 将test_A的真实结果(认为概率大于0.99的)
uidTest_A = pd.read_table('../data/uid_testA_answer5.txt', sep=',', header = 0)
#uidTest_B = pd.read_table('../data/uid_testB_answer2.txt', sep=',', header = 0)
uidTrain = pd.concat([uidTrain, uidTest_A])
#uidTrain = pd.concat([uidTrain, uidTest_B])

# 测试集A
uidTestA = pd.DataFrame()
uidTestA['uid'] = range(5000,7000)
uidTestA.uid = uidTestA.uid.apply(lambda x: 'u'+str(x).zfill(4))

# 测试集B
uidTestB = pd.DataFrame()
uidTestB['uid'] = range(7000,10000)
uidTestB.uid = uidTestB.uid.apply(lambda x: 'u'+str(x).zfill(4))

# 训练集
train = feature[feature.uid.isin(uidTrain.uid.values)].copy()
train = train.merge(uidTrain, on='uid', how='left')

testA = feature[4999:6999].copy()
#uidTestA = pd.DataFrame()
#uidTestA['uid'] = range(5000, 7000)
#uidTestA.uid = uidTestA.uid.apply(lambda x: 'u'+str(x).zfill(4))


# 打乱顺序
np.random.seed(201806)
idx = np.random.permutation(len(train))
train = train.iloc[idx]

X_train = train.drop(['uid','label'], axis=1).values
y_train = train.label.values
X_testA = testA.drop(['uid'], axis=1).values

# 测试集
testB = feature[6999:].copy()

X_testB = testB.drop(['uid'], axis=1).values

lgb1 = lightgbm.LGBMClassifier(random_state=201806)

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
          seed= 201806,
          nthread= 4,
          silent= True)


def fitModel(model, feature1):
    X = feature1.drop(['uid','label'], axis=1).values
    y = feature1.label.values
    
    lgb_y_proba = cross_val_predict(model, X, y, cv=5,
                                verbose=2, method='predict_proba')[:,1]
    
    lgb_y_proba1 = cross_val_predict(lgb1, X, y, cv=5,
                                    verbose=2, method='predict_proba')[:,1]
    
    lgb_y_proba = lgb_y_proba*0.55 + lgb_y_proba1*0.45
    
    lgb_y_pred = (lgb_y_proba >= 0.5)*1
    
    f1score = f1_score(y, lgb_y_pred)
    aucscore = roc_auc_score(y, lgb_y_proba)
    print('F1:', f1score,
          'AUC:', aucscore,
          'Score:', f1score*0.4 + aucscore*0.6)
    print('Num:', sum(lgb_y_pred))
    print(classification_report(y, lgb_y_pred))
    
    model.fit(X, y)

    featureList0 = list(feature1.drop(['uid','label'], axis=1))
    featureImportant = pd.DataFrame()
    featureImportant['feature'] = featureList0
    featureImportant['score'] = model.feature_importances_
    featureImportant.sort_values(by='score', ascending=False, inplace=True)
    featureImportant.reset_index(drop=True, inplace=True)
    print(featureImportant)
    
    return featureImportant


# 交叉验证模型
featureImportant = fitModel(lgb, train)

top_K = 150
X_train = train[featureImportant['feature'].values[:top_K]].values
X_testB = testB[featureImportant['feature'].values[:top_K]].values
X_testA = testA[featureImportant['feature'].values[:top_K]].values


lgb_y_proba1 = cross_val_predict(lgb1, X_train, y_train, cv=5,
                       verbose=2, method='predict_proba')[:,1]
lgb_y_proba = cross_val_predict(lgb, X_train, y_train, cv=5,
                            verbose=2, method='predict_proba')[:,1]

lgb_y_proba = lgb_y_proba*0.55 + lgb_y_proba1*0.45
lgb_y_pred = (lgb_y_proba >= 0.5)*1

f1score = f1_score(y_train, lgb_y_pred)
aucscore = roc_auc_score(y_train, lgb_y_proba)
print('F1:', f1score,
      'AUC:', aucscore,
      'Score:', f1score*0.4 + aucscore*0.6)
print(classification_report(y_train, lgb_y_pred))
print("LGB:", sum(lgb_y_pred))

train_with_proba = train[['uid','label']]
train_with_proba['proba'] = lgb_y_proba
train_with_proba['pred'] = lgb_y_pred
train_with_proba.sort_values('proba', ascending=False, inplace=True)

# LGB
# F1: 0.763783510369 AUC: 0.95628093573 Score: 0.879281965585
# F1: 0.776717557252 AUC: 0.967378883283 Score: 0.891114352871
# F1: 0.772875058059 AUC: 0.967378883283 Score: 0.889577353193


xgb = xgboost.XGBClassifier(random_state=201806, eval_metric='auc')
xgb_y_proba = cross_val_predict(xgb, X_train, y_train, cv=5,
                                verbose=2, method='predict_proba')[:,1]
xgb_y_pred = (xgb_y_proba >= 0.4)*1

f1score = f1_score(y_train, xgb_y_pred)
aucscore = roc_auc_score(y_train, xgb_y_proba)
print('F1:', f1score,
      'AUC:', aucscore,
      'Score:', f1score*0.4 + aucscore*0.6)
print(classification_report(y_train, xgb_y_pred))
print("XGB:", sum(xgb_y_pred))


# XGB
# F1: 0.709365558912 AUC: 0.953276679949 Score: 0.855712231534
# F1: 0.73063973064 AUC: 0.964762934967 Score: 0.871113653236
# F1: 0.75376344086 AUC: 0.964762934967 Score: 0.880363137324
# F1: 0.761904761905 AUC: 0.964762934967 Score: 0.883619665742

rf = RandomForestClassifier(random_state=201806)
rf_y_proba = cross_val_predict(rf, X_train, y_train, cv=5,
                               verbose=2, method='predict_proba')[:,1]
rf_y_pred = (rf_y_proba >= 0.4)*1

f1score = f1_score(y_train, rf_y_pred)
aucscore = roc_auc_score(y_train, rf_y_proba)
print('F1:', f1score,
      'AUC:', aucscore,
      'Score:', f1score*0.4 + aucscore*0.6)
print(classification_report(y_train, rf_y_pred))
print("RF:", sum(rf_y_pred))

# RF
# F1: 0.668584579977 AUC: 0.916614214849 Score: 0.8174023609
# F1: 0.705505077499 AUC: 0.932585884187 Score: 0.841753561512
# F1: 0.705505077499 AUC: 0.932585884187 Score: 0.841753561512
# F1: 0.717142857143 AUC: 0.932585884187 Score: 0.846408673369

gbdt = GradientBoostingClassifier(random_state=201806)
gbdt_y_proba = cross_val_predict(gbdt, X_train, y_train, cv=5,
                                 verbose=2, method='predict_proba')[:,1]
gbdt_y_pred = (gbdt_y_proba >= 0.4)*1

f1score = f1_score(y_train, gbdt_y_pred)
aucscore = roc_auc_score(y_train, gbdt_y_proba)
print('F1:', f1score,
      'AUC:', aucscore,
      'Score:', f1score*0.4 + aucscore*0.6)
print(classification_report(y_train, gbdt_y_pred))
print("GBDT:", sum(gbdt_y_pred))

# MIX
mix_y_proba = (lgb_y_proba + xgb_y_proba + rf_y_proba + gbdt_y_proba)/4.0
mix_y_pred = (mix_y_proba >= 0.4)*1

f1score = f1_score(y_train, mix_y_pred)
aucscore = roc_auc_score(y_train, mix_y_proba)
print('F1:', f1score,
      'AUC:', aucscore,
      'Score:', f1score*0.4 + aucscore*0.6)
print(classification_report(y_train, mix_y_pred))
print("MIX:", sum(mix_y_pred))

# GBDT
# F1: 0.710763680096 AUC: 0.952901249627 Score: 0.856046221815
# F1: 0.735785953177 AUC: 0.963951494146 Score: 0.872685277758
# F1: 0.757074212493 AUC: 0.963951494146 Score: 0.881200581485
# F1: 0.77153171738 AUC: 0.963951494146 Score: 0.88698358344

isXX = False
if isXX == True:
    lgb.fit(X_train, y_train)
    lgb1.fit(X_train, y_train)
    
    y_proba = lgb.predict_proba(X_testA)[:,1]
    y_proba1 = lgb1.predict_proba(X_testA)[:,1]
    
    y_proba = y_proba*0.55 + y_proba1*0.45
    
    y_pred = (y_proba >= 0.5)*1
    # y_pred = lgb.predict(X_test)
    
    uidTestA['proba'] = y_proba
    uidTestA['label'] = y_pred
    uidTestA.sort_values(by='proba', ascending=False, inplace=True)
    # uidTest[['uid','label']].to_csv('../output/result_20180604A.csv', index=False, header = False)
    
    
    print(uidTestA.label.value_counts())
    
    #uidTestA[(uidTestA.proba >= 0.99)|(uidTestA.proba <= 0.01)][['uid','label']].\
    #    to_csv('../data/uid_testA_answer4.txt', index=False )
    
else:
    
    lgb.fit(X_train, y_train)
    lgb1.fit(X_train, y_train)
    
    y_proba = lgb.predict_proba(X_testB)[:,1]
    y_proba1 = lgb1.predict_proba(X_testB)[:,1]
    
    y_proba = y_proba*0.55 + y_proba1*0.45
    
    y_pred = (y_proba >= 0.45)*1
    # y_pred = lgb.predict(X_test)
    
    uidTestB['proba'] = y_proba
    uidTestB['label'] = y_pred
    uidTestB.sort_values(by='proba', ascending=False, inplace=True)
    # uidTest[['uid','label']].to_csv('../output/result_20180604A.csv', index=False, header = False)
    print(uidTestB.label.value_counts())
    
    #uidTestB[(uidTestB.proba >= 0.99)|(uidTestB.proba <= 0.01)][['uid','label']].\
    #    to_csv('../data/uid_testB_answer2.txt', index=False )

"""
y_proba = lgb.predict_proba(X_testA)[:,1]
y_proba1 = lgb1.predict_proba(X_testA)[:,1]

y_proba = y_proba*0.55 + y_proba1*0.45

y_pred = (y_proba >= 0.5)*1
uidTestA['proba'] = y_proba
uidTestA['label'] = y_pred
uidTestA.sort_values(by='proba', ascending=False, inplace=True)
print(uidTestA.label.value_counts())

uidTestA_answer = uidTestA[(uidTestA.proba >= 0.98)|(uidTestA.proba <= 0.02)]
uidTestA_answer[['uid','label']].to_csv('../data/uid_testA_answer2.txt', index=False)
"""

"""
# 线上 0.84551
# uidTest[['uid','label']].to_csv('../output/result_20180602A.csv', index=False, header = False)

F1: 0.766902119072 AUC: 0.955587541677 Score: 0.880113372635
Num: 1082

(150) 0.847886
F1: 0.771614192904 AUC: 0.959340760619 Score: 0.884250133533
             precision    recall  f1-score   support

          0       0.97      0.92      0.94      4099
          1       0.70      0.86      0.77       900

avg / total       0.92      0.91      0.91      4999

1101
0    2392
1     608

(150) 0.851817
F1: 0.77878643096 AUC: 0.967494744498 Score: 0.892011419083
             precision    recall  f1-score   support

          0       0.97      0.94      0.95      5037
          1       0.72      0.85      0.78       963

avg / total       0.93      0.92      0.93      6000

(XGB+200)

F1: 0.768846729905 AUC: 0.970615672586 Score: 0.889908095513
             precision    recall  f1-score   support

          0       0.96      0.96      0.96      5990
          1       0.76      0.78      0.77       990

avg / total       0.93      0.93      0.93      6980

XGB: 1013

xgb = xgboost.XGBClassifier(random_state=201806)
xgb_y_proba = cross_val_predict(xgb, X_train, y_train, cv=5,
                                verbose=2, method='predict_proba')[:,1]
xgb_y_pred = (xgb_y_proba >= 0.35)*1

f1score = f1_score(y_train, xgb_y_pred)
aucscore = roc_auc_score(y_train, xgb_y_proba)
print('F1:', f1score,
      'AUC:', aucscore,
      'Score:', f1score*0.4 + aucscore*0.6)
print(classification_report(y_train, xgb_y_pred))
print("XGB:", sum(xgb_y_pred))

xgb.fit(X_train, y_train)
xgb_y_proba = xgb.predict_proba(X_testB)[:,1]
xgb_y_pred = (xgb_y_proba >= 0.35)*1
uidTestB['proba'] = y_proba
uidTestB['label'] = y_pred
uidTestB.sort_values(by='proba', ascending=False, inplace=True)
# uidTest[['uid','label']].to_csv('../output/result_20180604A.csv', index=False, header = False)
print(uidTestB.label.value_counts())

"""









