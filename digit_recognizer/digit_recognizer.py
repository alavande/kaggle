import pandas as pd
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

def split_data(df):
    # label
    df_label = df.label
    # features
    df_features = df.drop("label", axis=1)

    return df_label, df_features

def nomalizing(df):
    for key in df.keys():
        df.loc[(df[key] != 0), key] = 1

    return df

train_label, train_features = split_data(train_data)

nomal_train_features = nomalizing(train_features)
nomal_test_features = nomalizing(test_data)

from sklearn.neighbors import KNeighborsClassifier as knn

clf = knn(n_neighbors=20, weights='distance',algorithm='auto', n_jobs=-1)

clf.fit(nomal_train_features, train_label)

prediction = clf.predict(nomal_test_features)

result = pd.DataFrame({"Label": prediction})

result.to_csv("knn_prediction.csv", index=False)


