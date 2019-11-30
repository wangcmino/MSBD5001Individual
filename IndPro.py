#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import xgboost as xgb
from datetime import timedelta
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from datetime import datetime
CSV_FILE_PATH = 'E:/Study/BDT/MSBD5001/msbd5001-fall2019/train.csv'
CSV_FILE_PATH1 = 'E:/Study/BDT/MSBD5001/msbd5001-fall2019/test.csv'
CSV_FILE_PATH2 = 'E:/Study/BDT/MSBD5001/msbd5001-fall2019/samplesubmission.csv'

def TimeTransform(fm, t):
    month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                         'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    for month,value in month_map.items():
       t =  t.replace(month, value)
    return  datetime.datetime.strptime(t, fm)

def DataProcess(df,flag):
    print("processing-----")
    df['is_free'] = df['is_free'].astype('int')
    #fill the missing values
    df['total_positive_reviews'].fillna(df['total_positive_reviews'].mean(), inplace = True)
    df['total_negative_reviews'].fillna(df['total_negative_reviews'].mean(), inplace=True)

    # now_date = 'Aug 11, 2019'
    # df['purchase_date'].fillna(now_date,inplace=True)
    #
    # # time data transform
    # df['purchase_date'] = df['purchase_date'].apply(lambda it: TimeTransform(fm='%m %d, %Y',t = it))
    # if flag ==1 :
    #   df['release_date'].fillna('11 Aug, 2019', inplace=True)
    #   df['release_date'] = df['release_date'].apply(lambda it: TimeTransform(fm='%d %m, %Y', t=it))
    # else:
    #     df['release_date'].fillna('11-Aug-19', inplace=True)
    #     df['release_date'] = df['release_date'].apply(lambda it: TimeTransform(fm='%d-%m-%y', t=it))
    # df['p_sub_r'] = ((df['purchase_date']-df['release_date'])/np.timedelta64(1, 'D')).astype(int)
    # df.p_sub_r[df['p_sub_r'] < 0] = 0
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['release_date'] = pd.to_datetime(df['release_date'])
    for i in range(df.shape[0]):
        if(df.purchase_date[i] - df.release_date[i]).days < 0:
            t = df.purchase_date[i]
            df.purchase_date[i] = df.release_date[i]
            df.release_date[i] = t
    df['p_sub_r'] = (df['purchase_date']-df['release_date']).astype('timedelta64[D]')
    df['p_sub_r'].fillna(df['p_sub_r'].mean(),inplace=True)

    df['purchase_date'].fillna(df['release_date']+timedelta(days=df['p_sub_r'].mean()),inplace= True)
    now_date = datetime(2019, 10, 31)
    df['pur_delta'] = ((now_date - df['purchase_date'])/np.timedelta64(1, 'D')).astype(int)
    df['rel_delta'] = ((now_date - df['release_date'])/np.timedelta64(1, 'D')).astype(int)

    #encode the categorical data using onehote coding
    genres = df['genres'].str.get_dummies(sep=',')
    cate = df['categories'].str.get_dummies(sep=',')
    tags =df['tags'].str.get_dummies(sep=',')
    DF = pd.concat([df,genres,cate,tags],axis=1)
    # check the NAN values
    #print(DF.isnull().sum().sort_values(ascending=False))



    return DF

def TrainTestDict(train, test):

    traincv = train.columns.values.tolist()
    testcv = test.columns.values.tolist()
    train_only = pd.DataFrame(index= range(len(test)))
    test_only = pd.DataFrame(index= range(len(train)))
    for i in range(len(traincv)):
         if traincv[i] not in testcv:
             train_only[traincv[i]] = '0'

    for j in range(len(testcv)):
        if testcv[j] not in traincv:
            test_only[testcv[j]] = '0'
    print(test_only.shape,train.shape,train_only.shape,test.shape)
    train.to_csv('ori_train.csv',index=True)
    test.to_csv('ori_test.csv', index=True)
    test_only.to_csv('test_only.csv', index=True)
    train_only.to_csv('train_only.csv', index=True)
    train = pd.concat([train,test_only],axis=1,join_axes=[train.index])
    test = pd.concat([test,train_only],axis=1,join_axes=[test.index])

    print(train.shape,test.shape)
    # preserve the unduplicated columns
    Train = train.loc[:,~train.columns.duplicated()]
    Test = test.loc[:,~test.columns.duplicated()]
    print(Train.shape,Test.shape)
    # print(train.isnull().sum())
    # print(test.isnull().sum())
    return Train,Test

def embeding_featrue(train,test):
    print("-----row:",train.shape[0],test.shape[0])
    #drop feature note exist in train
    test = test[list(train.columns)]
    y = np.array(train['playtime_forever'])
    droplist = ['playtime_forever','id', 'is_free', 'genres', 'categories','tags','purchase_date','release_date']
    feature = train.drop(droplist,axis=1)
    feature_test = test.drop(droplist,axis=1)
    feature_list = list(feature.columns)
    scols = ['p_sub_r','total_positive_reviews', 'total_negative_reviews','price','pur_delta','rel_delta']
    scaler = MinMaxScaler()
    feature[scols] = scaler.fit_transform(feature[scols])
    feature[scols] = scaler.transform(feature[scols])

    feature_test[scols] = scaler.fit_transform(feature_test[scols])
    feature_test[scols] = scaler.transform(feature_test[scols])

    feature = np.array(feature)
    feature_test = np.array((feature_test))

    #print(feature.shape,feature_test.shape,feature_list)
    return feature,y,feature_test

def RF_fit(feaure,y):
    rf = rfr(n_estimators = 1000,random_state = 42)
    rf.fit(feaure,y)
    return rf
def xgboost_predictor(xtrain,xtest,y):

    xgbmodel = xgb.XGBRegressor(
        max_depth=5,
        n_estimators=200,
        learning_rate=0.01,
        nthread=4,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_=3,
        silent=0)

    # fit the model
    xgbmodel.fit(xtrain, y)
    kfold = KFold(n_splits=5,shuffle=False)
    score = cross_val_score(xgbmodel,xtrain, y, cv=kfold, scoring='neg_mean_squared_error')
    print("Cross Validation score for Random Forest: ", "RMSEtrain: ", score)
    #predict
    predict_y = xgbmodel.predict(xtest)
    return predict_y

train = pd.read_csv(CSV_FILE_PATH)
test = pd.read_csv(CSV_FILE_PATH1)


train=DataProcess(train,flag = 1)
test=DataProcess(test,flag = 0)

train,test = TrainTestDict(train,test)
feature,y,feature_test=embeding_featrue(train,test)
# rfmodel = RF_fit(feature,y)
# predict_y = rfmodel.predict(feature)
# kfold = KFold(n_splits=5,shuffle=False)
# score = cross_val_score(rfmodel,feature, y, cv=kfold, scoring='neg_mean_squared_error')
# print("Cross Validation score for Random Forest: ", "RMSEtrain: ", score)
# Test['playtime_forever'] = rfmodel.predict(feature_test)
#np.savetxt("temp.csv", feature, delimiter=",")
predict = xgboost_predictor(feature,feature_test,y)
predict[predict<0] = 0

sub_train = train.loc[(train['playtime_forever'] > 1)]
sub_feature,sub_y,feature_test=embeding_featrue(sub_train,test)
sub_predict =  xgboost_predictor(sub_feature,feature_test,sub_y)
for i in range(len(predict)):
    if predict[i] > 5:
        predict[i] = sub_predict[i]

test['playtime_forever'] = predict
output = test[['id','playtime_forever']]
print(output.dtypes)
output.to_csv(CSV_FILE_PATH2, index = False)



