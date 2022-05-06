import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Load Data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
x_train = df_train.iloc[:,2:]
y_train = df_train.y
x_test = df_test.iloc[:,1:]

# Train
# model = linear_model.LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

# Evaluation
y_mean = x_test.mean(axis = 1) 
# RMSE = mean_squared_error(y_mean, y_pred)**0.5
# print(RMSE)

# Output
df_sub = pd.read_csv('./data/sample.csv')
id_sub = df_sub.Id
sub = pd.DataFrame({'Id':id_sub, 'y':y_mean})
sub.to_csv('sub2.csv', index = False)