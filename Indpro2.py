#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import xgboost as xgb
from datetime import timedelta
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.manifold import SpectralEmbedding
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,cross_validation,ensemble
from datetime import datetime
CSV_FILE_PATH = 'E:/Study/BDT/MSBD5001/msbd5001-fall2019/train.csv'
CSV_FILE_PATH1 = 'E:/Study/BDT/MSBD5001/msbd5001-fall2019/test.csv'
CSV_FILE_PATH2 = 'E:/Study/BDT/MSBD5001/msbd5001-fall2019/emrfsubmission.csv'

def TimeTransform(fm, t):
    month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                         'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    for month,value in month_map.items():
       t =  t.replace(month, value)
    return  datetime.datetime.strptime(t, fm)

def DataProcess(df):
    print("processing-----")
    df['is_free'] = df['is_free'].astype('int')
    #fill the missing values
    df['total_positive_reviews'].fillna(df['total_positive_reviews'].mean(), inplace = True)
    df['total_negative_reviews'].fillna(df['total_negative_reviews'].mean(), inplace=True)

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
    now_date = datetime(2019, 8, 31)
    df['pur_delta'] = ((now_date - df['purchase_date'])/np.timedelta64(1, 'D')).astype(int)
    df['rel_delta'] = ((now_date - df['release_date'])/np.timedelta64(1, 'D')).astype(int)

    df['release_year'] = df["release_date"].apply(lambda dt: dt.year)
    df['release_month'] = df["release_date"].apply(lambda dt: dt.month)
    df['release_day'] = df["release_date"].apply(lambda dt: dt.day)

    df['purchase_year'] = df["purchase_date"].apply(lambda dt: dt.year)
    df['purchase_month'] = df["purchase_date"].apply(lambda dt: dt.month)
    df['purchase_day'] = df["purchase_date"].apply(lambda dt: dt.day)
    #encode the categorical data using onehote coding
    genres = df['genres'].str.get_dummies(sep=',')
    cate = df['categories'].str.get_dummies(sep=',')
    tags =df['tags'].str.get_dummies(sep=',')
    DF = pd.concat([df,genres,cate,tags],axis=1)
    droplist = ['id', 'is_free', 'genres', 'categories', 'tags', 'purchase_date', 'release_date']
    DF = DF.drop(droplist, axis=1)
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

    train = pd.concat([train,test_only],axis=1,join_axes=[train.index])
    test = pd.concat([test,train_only],axis=1,join_axes=[test.index])

    print(train.shape,test.shape)
    # preserve the unduplicated columns
    Train = train.loc[:,~train.columns.duplicated()]
    Test = test.loc[:,~test.columns.duplicated()]
    print(Train.shape,Test.shape)

    return Train,Test

def Nomalization(df):


    scols = ['p_sub_r','total_positive_reviews', 'total_negative_reviews',\
             'price','pur_delta','rel_delta',\
             'purchase_year','purchase_month','purchase_day',\
             'release_year','release_month','release_day']

    scaler = MinMaxScaler()
    df[scols] = scaler.fit_transform(df[scols])
    df[scols] = scaler.transform(df[scols])
    df = df.loc[:, ~df.columns.duplicated()]

    return df


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
    kfold = KFold(n_splits=5, shuffle=False)
    score = cross_val_score(xgbmodel, xtrain, y, cv=kfold, scoring='neg_mean_squared_error')
    print("Cross Validation score for Random Forest: ", "RMSEtrain: ", score)
    # fit the model
    xgbmodel.fit(xtrain, y)
    #predict
    predict_y = xgbmodel.predict(xtest)
    return predict_y

# main function

train = pd.read_csv(CSV_FILE_PATH)
test = pd.read_csv(CSV_FILE_PATH1)
train_len = train.shape[0]

totaldf = train.append(test,ignore_index = True)
totaldf=DataProcess(totaldf)
Nom_df=Nomalization(totaldf)

Trainy = Nom_df[:train_len]
Test =Nom_df[train_len:]

y_train = Trainy['playtime_forever']
Train = Trainy.drop(['playtime_forever'],axis=1,inplace=False)
embedding = SpectralEmbedding(n_components=50)
emTrain = embedding.fit_transform(Train)

Test.drop(['playtime_forever'],axis=1,inplace=True)
embedding = SpectralEmbedding(n_components=50)
emTest = embedding.fit_transform(Test)
print(emTrain.shape,emTest.shape)

# predict = xgboost_predictor(emTrain,emTest,y_train)

RFmodel = ensemble.RandomForestRegressor(n_estimators=100)
score_rf = cross_val_score(RFmodel, emTrain, y_train, cv=3, scoring='neg_mean_squared_error')
print("Cross Validation score for Random Forest: ", "MSE: ", score_rf)
RFmodel.fit(emTrain,y_train)
print("Traing Score:%f"%RFmodel.score(emTrain,y_train))
predict = RFmodel.predict(emTest)

# par = PassiveAggressiveRegressor()
# kfold = KFold(n_splits=5, shuffle=False)
# score = cross_val_score(par, emTrain, y_train, cv=kfold, scoring='neg_mean_squared_error')
# print("Cross Validation score for PassiveAggressiveRegressor: ", "RMSEtrain: ", score)
# par.fit(emTrain,y_train)
# predict = par.predict(emTest)

predict[predict<0] = 0
sub_train = Trainy.loc[(Trainy['playtime_forever'] > 1)]
sub_y = sub_train['playtime_forever']
sub_train.drop(['playtime_forever'],axis=1,inplace=True)
embedding = SpectralEmbedding(n_components=50)
emsub_train = embedding.fit_transform(sub_train)

RFmodel2 = ensemble.RandomForestRegressor(n_estimators=100)
score_rf2 = cross_val_score(RFmodel2, sub_train, sub_y, cv=3, scoring='neg_mean_squared_error')
print("Cross Validation score for Random Forest: ", "MSE: ", score_rf2)
RFmodel2.fit(sub_train,sub_y)
print("subTraing Score:%f"%RFmodel2.score(sub_train,sub_y))
sub_predict = RFmodel2.predict(Test)

# sub_predict =  xgboost_predictor(emsub_train,emTest,sub_y)

# par2 = PassiveAggressiveRegressor()
# kfold = KFold(n_splits=5, shuffle=False)
# score = cross_val_score(par, emsub_train, sub_y, cv=kfold, scoring='neg_mean_squared_error')
# print("Cross Validation score for PassiveAggressiveRegressor: ", "RMSEtrain: ", score)
# par2.fit(emsub_train,sub_y)
# sub_predict = par2.predict(emTest)


for i in range(len(predict)):
    if predict[i] > 5:
        predict[i] = sub_predict[i]

test['playtime_forever'] = predict
output = test[['id','playtime_forever']]
print(output.dtypes)
output.to_csv(CSV_FILE_PATH2, index = False)



