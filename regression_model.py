import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from performance_statistics import load_overall_statistics_x_y
import scipy.stats as scs
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor

def stats_model(X,y):
    est = sm.OLS(y,X).fit()
    est.summary()

def sklearn_Linear_Regression(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return y_pred

def ensemble_methods(X_train,X_test, y_train,y_test):
    rf = RandomForestRegressor()


    gdbr = GradientBoostingRegressor(learning_rate=0.01, loss='ls',
                                     n_estimators=10000, random_state=1)

    abr = AdaBoostRegressor(DecisionTreeRegressor(), learning_rate=0.1,
                            loss='linear', n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    gdbr.fit(X_test, y_test)
    abr.fit(X_train, y_train)

def evaluate_model(model, X_train, y_train):
    '''
    INPUT
         - model: this is a classification model from sklearn
         - X_train: 2d array of the features
         - y_train: 1d array of the target
    OUTPUT
         - information about the model's accuracy using 10
         fold cross validation
         - model: the fit model
    Returns the model
    '''
    print(np.mean(cross_val_score(model, X_train, y_train,
                                  cv=10, n_jobs=-1, verbose=10)))
    model.fit(X_train, y_train)
    return model

def gridsearch(paramgrid, X_train, y_train):
    rf = RandomForestRegressor(n_estimators=200)
    gridsearch = GridSearchCV(rf,
                              paramgrid,
                              n_jobs=-1,
                              verbose=10,
                              cv=10,
                              scoring=rlmse_gs)
    gridsearch.fit(X_train, y_train)
    best_model = gridsearch.best_estimator_
    print('these are the parameters of the best model')
    print(best_model)
    print('\nthese is the best score')
    print(gridsearch.best_score_)
    return best_model, gridsearch

def rlmse(y_pred,y_test):
    target = y_test
    predictions = y_pred
    log_diff = np.log(predictions + 1) - np.log(target + 1)
    rmse = np.sqrt(np.mean(log_diff**2))
    return rmse

def rlmse_gs(model, X_test,y_test):
    target = y_test
    predictions = model.predict(X_test)
    log_diff = np.log(predictions + 1) - np.log(target + 1)
    rmse = np.sqrt(np.mean(log_diff**2))
    return rmse

def view_feature_importances(df, model):
    '''
    INPUT
         - df: dataframe which has the original data
         - model: this is the sklearn classification model that has
         already been fit (work with tree based models)
    OUTPUT
         - prints the feature importances in descending order
    Returns nothing
    '''
    columns = df.columns
    features = model.feature_importances_
    featimps = []
    for column, feature in zip(columns, features):
        featimps.append([column, feature])
    print(pd.DataFrame(featimps, columns=['Features',
                       'Importances']).sort_values(by='Importances',
                                                   ascending=False))
def scatter_plot(X,y):
    for column in X.columns:
        plt.scatter(X[column].values,y)
        plt.xlabel(column)
        #plt.ylabel(Salary)
        plt.show()

if __name__ == '__main__':
    overall_statistics, X, y, batting_auction_data, bowling_auction_data = load_overall_statistics_x_y()
    #df_train = overall_statistics[overall_statistics.season.isin([2013,2014,2015])]

    df_train = overall_statistics[overall_statistics.season != 2016]
    y_train = df_train['Salary']
    X_train = df_train.drop(['Player','Salary'],axis=1)

    df_test = overall_statistics[overall_statistics.season == 2016]
    y_test = df_test['Salary']
    X_test = df_test.drop(['Player','Salary'],axis=1)
    '''
    df_train = batting_auction_data[batting_auction_data != 2016]
    X_train = df_train.drop(['Player','batsman','Salary'],axis=1)
    y_train = df_train['Salary']

    df_test = batting_auction_data[batting_auction_data.season == 2016]
    X_test = df_test.drop(['Player','batsman','Salary'],axis=1)
    y_test = df_test['Salary']
    '''
    #y_pred_LR = sklearn_Linear_Regression(X_train, X_test, y_train, y_test)
    paramgrid = dict(max_depth=[5,10,20,30],min_samples_split=[12, 16, 20],min_samples_leaf=[21, 25, 29, 33])

    best_model, gridsearch = gridsearch(paramgrid, X_train, y_train)
    print "Best rlmse for test model is"
    print rlmse_gs(best_model,X_test,y_test)
    print "Best score for train model is"
    print best_model.score(X_train,y_train)
    print "Best model score is"
    print best_model.score(X_test,y_test)
    view_feature_importances(X, best_model)
    '''
    svr = SVR(C = 0.1)
    svr.fit(X_train, y_train)
    svr.predict(X_test)
    '''
