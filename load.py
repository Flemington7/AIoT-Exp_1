import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing

warnings.filterwarnings("ignore", category = FutureWarning)  # ignore the warning

train_csv = "train.csv"
train_data = pd.read_csv(train_csv)

all_labels_train = np.array(train_data['actual'])  # pd -> np
features_train = train_data.drop('actual', axis = 1)  # axis=1 means drop column
features_train = pd.get_dummies(features_train)  # one-hot

all_features_train = preprocessing.StandardScaler().fit_transform(features_train)  # standardization, pd -> np

def getdata(clients_num):  # shuffle data, IID
    index_order = np.arange(all_features_train.shape[0])  # get the index of all data
    np.random.shuffle(index_order)  # shuffle the index
    ordered_index = np.array_split(index_order,
                                   clients_num)  # balance split the index into clients_num parts, ordered_index is a
    # array
    return ordered_index
    # all_features_train_shuffle = all_features_train[index_order]#shuffle the data
    # all_labels_train_shuffle = all_labels_train[index_order]#shuffle the label
