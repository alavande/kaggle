# coding=utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame as df

data_train = pd.read_csv("train.csv")

# 查看信息，显示特征名称，不为空数据数，数据格式，如：
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
#print data_train.info()

# 只能得到数值型数据分布，部分特征会缺失
#print data_train.describe()

# import 随机森林
from sklearn.ensemble import RandomForestRegressor as rfr
def set_missing_ages(df):
    # 取出已有的数值型特征
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 把乘客分成已知年龄和未知年龄
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # 训练集
    y = known_age[:, 0]
    X = known_age[:, 1:]

    regressor = rfr(random_state=0, n_estimators=2000, n_jobs=-1)
    regressor.fit(X,y)

    return regressor

def replace_age(df, regressor):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # 用得到的模型进行未知年龄结果预测
    predictedAges = regressor.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df

def set_Cabin_and_Sex(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "1"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "0"
    df.loc[(df.Sex == 'female'), 'Sex'] = "1"
    df.loc[(df.Sex == 'male'), 'Sex'] = "0"
    return df

import sklearn.preprocessing as pre
def preprocess_data(df, regressor):

    df = replace_age(df, regressor)
    df = set_Cabin_and_Sex(df)

    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df = pd.concat([df, dummies_Embarked, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Embarked'], axis=1, inplace=True)

    scaler = pre.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    df.drop(['Age', 'Fare'], axis=1, inplace=True)
    df.drop(['Name', 'Ticket'], axis=1, inplace=True)
    return df

regressor = set_missing_ages(data_train)
data_train = preprocess_data(data_train, regressor)

from sklearn import linear_model as lm

train_label = data_train.Survived
train_feature = data_train.drop(['PassengerId', 'Survived'], axis=1)

clf = lm.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_feature, train_label)

data_test = pd.read_csv("test.csv")
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
data_test = preprocess_data(data_test, regressor)

passenger = data_test.PassengerId
predictions = clf.predict(data_test.drop("PassengerId", axis=1))

result = pd.DataFrame({'PassengerId':passenger.as_matrix(), 'Survived':predictions.astype(np.int32)})

# result.to_csv("lr_predictions.csv", index=False)


answer = pd.read_csv("gender_submission.csv")

length = result.shape[0]
count = 0
for i in range(length):
    if result.Survived[i] == answer.Survived[i]:
        count = count + 1.0

print count / length