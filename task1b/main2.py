import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import sklearn.model_selection as ms

df_train = pd.read_csv('./data/train.csv');
x = df_train.iloc[:,2:]
y = df_train.y

x_train = np.array(x)
y_train = np.array(y)
x_train
# clf = Ridge(alpha= 0.1)

x_quadratic = np.power(x_train, 2)
x_exponential = np.exp(x_train)
x_cos = np.cos(x_train)
const = np.ones(700).reshape((-1,1))
x_all = np.concatenate((x_train, x_quadratic, x_exponential, x_cos, const), axis=1)

# k fold cv
lamda_set = np.array([0.1,10,20,30,100])
RMSE = np.zeros(5)
cv_ = ms.KFold(n_splits=10, shuffle=True)

i = 0
for lamda in lamda_set:
    clf = Ridge(alpha= lamda, fit_intercept=False)
    scores = ms.cross_val_score(clf, x_all, y_train, cv=cv_, scoring='neg_root_mean_squared_error')
    RMSE[i] = -scores.mean()
    i=i+1

best_lamda_index = np.where(RMSE == np.min(RMSE))
# print(lamda_set[best_lamda_index])

clf_all = Ridge(alpha=lamda_set[best_lamda_index], fit_intercept=False)
clf_all.fit(x_all, y_train)

output = pd.DataFrame(data = clf_all.coef_)
output.to_csv('sub5.csv', index = False, header = None)