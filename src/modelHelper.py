from __future__ import print_function
import numpy as np
import pandas as pd
import sklearn.discriminant_analysis
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
from sklearn.grid_search import GridSearchCV
    
def addFeatures(dataframe, adjclose, returns, n):
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = dataframe[returns].rolling(12).mean()
    exp_ma = returns[7:] + "ExponentMovingAvg" + str(n)
    dataframe.ewm(ignore_na=False,span=30.007751938,min_periods=0,adjust=True).mean()
    
def applyTimeLag(dataset, lags, delta):
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        newcolumn = column + str(maxLag)
        dataset[newcolumn] = dataset[column].shift(maxLag)

    return dataset.iloc[maxLag:-1, :]

def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy

# REGRESSION

def performRegression(dataset, split, symbol, output_dir):

    features = dataset.columns[1:]
    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]
    print('Size of train set: ', train.shape)
    print('Size of test set: ', test.shape)
    out_params = (symbol, output_dir)
    output = dataset.columns[0]
    predicted_values = []
    classifiers = [
        RandomForestRegressor(n_estimators=10, n_jobs=-1),
        SVR(C=100000, kernel='rbf', epsilon=0.1, gamma=1, degree=2),
        BaggingRegressor(),
        AdaBoostRegressor(),
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
    ]
    for classifier in classifiers:
        predicted_values.append(benchmark_model(classifier, train, test, features, output, out_params))
    maxiter = 1000
    batch = 150
    print('-'*80)
    mean_squared_errors = []
    r2_scores = []
    for pred in predicted_values:
        mean_squared_errors.append(mean_squared_error(test[output], pred))
        r2_scores.append(r2_score(test[output], pred))
    print(mean_squared_errors, r2_scores)
    return mean_squared_errors, r2_scores

def benchmark_model(model, train, test, features, output, output_params, *args, **kwargs):
    print('-'*80)
    model_name = model.__str__().split('(')[0].replace('Regressor', ' Regressor')
    print(model_name)
    symbol, output_dir = output_params
    model.fit(train[features].values, train[output].values, *args, **kwargs)
    predicted_value = model.predict(test[features].values)
    plt.plot(test[output].values, color='g', ls='-', label='Actual Value')
    plt.plot(predicted_value, color='b', ls='--', label='predicted Value')
    plt.xlabel('No of Set')
    plt.ylabel('Output Val')
    plt.title(model_name)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(symbol) + '_' + model_name + '.png'), dpi=100)
    plt.clf()
    return predicted_value