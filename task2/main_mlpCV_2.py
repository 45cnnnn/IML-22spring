from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

df_train_features = pd.read_csv('./data/train_features.csv')
df_train_labels = pd.read_csv('./data/train_labels.csv')
df_test_features = pd.read_csv('./data/test_features.csv')

#---------------------- Pre-processing ----------------------#
# scaler = StandardScaler()
# # scaler = RobustScaler()
# imp = SimpleImputer(strategy='mean')
def preprocessing_data(df_train, df_test, impute_method, standardize_method, normalize_method):
    #imputation of missing values
    if impute_method == 'mean':
        imp = SimpleImputer(strategy='mean')
    elif impute_method == 'median':
        imp = SimpleImputer(strategy='median')
    elif impute_method == 'KNN':
        imp = KNNImputer(n_neighbors=5, weights='distance')

    if standardize_method == 'standardscaler':
        scaler = StandardScaler()
    elif standardize_method == 'robustscaler':
        scaler = RobustScaler()
    
    if normalize_method == 'l2':
        normalizer = Normalizer(norm='l2')
    elif normalize_method == 'max':
        normalizer = Normalizer(norm='max')

    data_train = df_train.groupby('pid', sort=False).agg([np.nanmax, np.nanmin, np.nanmean, np.nanmedian, np.nanstd, np.nanvar])
    data_train = data_train.drop(columns='Time')
    data_train = imp.fit_transform(data_train)
    data_train = scaler.fit_transform(data_train)
    # data = normalizer.fit_transform(data)

    data_test = df_test.groupby('pid', sort=False).agg([np.nanmax, np.nanmin, np.nanmean, np.nanmedian, np.nanstd, np.nanvar])
    data_test = data_test.drop(columns='Time')
    data_test = imp.transform(data_test)
    data_test = scaler.transform(data_test)

    
    return data_train, data_test

x_train, x_test = preprocessing_data(df_train_features, df_test_features, 'median', 'standardscaler', 'l2')
# x_test = preprocessing_data(df_test_features, 'median', 'standardscaler', 'l2')
y_pred = pd.DataFrame({'pid':df_test_features['pid'].drop_duplicates()})

#TODO: Try other methods
#---------------------- Sub Task 1 ----------------------#
label1 = df_train_labels.columns[1:11]

parameters = {'alpha':[1.0,1.5,2.0,2.5,3.0]}
# parameters = np.array([1.5, 2.0, 2.5, 2.0, 2.5, 2.0, 1.0, 1.5, 1.5, 1.0])

# for i in range(len(label1)):
for i in tqdm(range(len(label1))):
    y_train = df_train_labels[label1[i]]
    # clf1 = svm.SVC(probability=True, kernel='sigmoid')
    # clf1 = svm.SVC(probability=True)
    # scores = ms.cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')
    
    mlp1 = MLPClassifier(max_iter=500, n_iter_no_change=30, verbose=False, learning_rate='adaptive')
    clf1  = GridSearchCV(estimator=mlp1, param_grid=parameters, cv=5, scoring='roc_auc', n_jobs=-1)
    
    # clf1 = MLPClassifier(max_iter=500, n_iter_no_change=30, verbose=False, learning_rate='adaptive', alpha=parameters[i])
    clf1.fit(x_train, y_train)
    pred = clf1.predict_proba(x_test)
    y_pred[label1[i]] = np.array(pred[:,1])
    # print(label1[i], ': ', clf1.best_params_)

#---------------------- Sub Task 2 ----------------------#
label2 = df_train_labels.columns[11]
y_train = df_train_labels[label2]

mlp2 = MLPClassifier(max_iter=500, n_iter_no_change=30, verbose=False, learning_rate='adaptive')
clf2  = GridSearchCV(estimator=mlp2, param_grid=parameters, cv=5, scoring='roc_auc', n_jobs=-1)

# clf2 = MLPClassifier(max_iter=500, n_iter_no_change=30, verbose=False, learning_rate='adaptive', alpha=2.0)
clf2.fit(x_train, y_train)
pred = clf2.predict_proba(x_test)
y_pred[label2] = np.array(pred[:,1])
# print(label2, ': ', clf2.best_params_)

#---------------------- Sub Task 3 ----------------------#
parameters2  = {'alpha':[5, 5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]}
# parameters2 = np.array([10.0, 9.5, 8.5, 10.0])
label3 = df_train_labels.columns[12:]

# for i in range(len(label3)):
for i in tqdm(range(len(label3))):
    y_train = df_train_labels[label3[i]]
    # regr = svm.SVR()
    
    mlp3 = MLPRegressor(max_iter=500, n_iter_no_change=30, verbose=False, learning_rate='adaptive')
    regr  = GridSearchCV(estimator=mlp3, param_grid=parameters2, cv=5, scoring='r2', n_jobs=-1)
    # regr = MLPRegressor(max_iter=500, n_iter_no_change=30, verbose=False, learning_rate='adaptive', alpha=parameters2[i])
    regr.fit(x_train, y_train)
    pred = regr.predict(x_test)
    y_pred[label3[i]] = pred
    print(label3[i], ': ', regr.best_params_)


#---------------------- Output ----------------------#
y_pred.to_csv('prediction8.zip', index=False, float_format='%.3f', compression='zip')
y_pred.to_csv('prediction8.csv', index = False, float_format='%.3f')