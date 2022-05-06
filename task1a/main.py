import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

df_train = pd.read_csv('./data/train.csv' );
x = df_train.iloc[:,1:]
y = df_train.y
kf = KFold(n_splits=10)
# kf.get_n_splits(x_train)

lambda_list = [0.1, 1, 10, 100, 200]
RMSE_list = []


for lambda_ in lambda_list:
    clf = Ridge(alpha= lambda_)
    RMSE = 0
    for train_index, test_index in kf.split(x):
        # print("Train:", train_index, "Test: ", test_index)
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        RMSE = RMSE + mean_squared_error(y_test, y_pred)**0.5
        # print(RMSE)
    RMSE = RMSE / 10.0
    RMSE_list.append(RMSE)

output = pd.DataFrame(data = RMSE_list)
output.to_csv('sub2.csv', index = False, header = None)


