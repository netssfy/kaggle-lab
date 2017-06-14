'data engineering'
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
from xgboost.sklearn import XGBRegressor

pd.set_option('display.float_format', lambda x: '%.6f'%x)

def GetXGBRegressor(X, Y):
    print('=' * 10 + 'XGBRegressor' + '=' * 10)
    # etas = [0.01, 0.03, 0.1, 0.2, 0.3, 0.6]
    # depths = [3, 4, 5, 6, 7, 8, 9, 10]
    # child_weights = [1, 2, 3]
    # gammas = [0.01, 0.03, 0.1, 0.3]
    # params = []
    # for a1 in etas:
    #     for a2 in depths:
    #         for a3 in child_weights:
    #             for a4 in gammas:
    #                 params.append({ 'eta': a1, 'depth': a2, 'child_weight': a3, 'gamma': a4})

    # X = X.as_matrix()
    # bestScore = 10000000
    # bestParam = None
    # bestModel = None
    # for param in params:
    #     print('running ' + str(param))
    #     model = XGBRegressor(learning_rate=param['eta'], max_depth=param['depth'], min_child_weight=param['child_weight'], gamma=param['gamma'])
    #     scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=3)
    #     score = np.sqrt(-scores).mean()

    #     print('score=%f'%(score))
    #     if score < bestScore:
    #         bestScore = score
    #         bestParam = param
    #         bestModel = model
    param = {'child_weight': 2, 'depth': 4, 'eta': 0.1, 'gamma': 0.01}
    print('best param = ' + str(param))
    model = XGBRegressor(learning_rate=param['eta'], max_depth=param['depth'], min_child_weight=param['child_weight'], gamma=param['gamma'])
    X = X.as_matrix()
    model.fit(X, Y)
    print('=' * 10 + 'XGBRegressor' + '=' * 10)
    return model;

def GetRandomForestRegressor(X, Y):
    print('=' * 10 + 'RandomForestRegressor' + '=' * 10)
    narray = [10, 30, 100]
    bestScore = 100000000
    bestN = 10
    bestModel = None
    for n in narray:
        model = RandomForestRegressor(n_estimators=n)
        scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=10)
        score = np.sqrt(-scores).mean()
        print('score=%f at n=%d'%(score, n))
        if score < bestScore:
            bestScore = score
            bestN = n
            bestModel = model
    
    print('best n=%d best score=%f'%(bestN, bestScore))
    print('=' * 10 + 'RandomForestRegressor' + '=' * 10)
    bestModel.fit(X, Y)
    return bestModel

def GetRidgeCV(dataX, dataY):
    print('=' * 10 + 'RidgeCV' + '=' * 10)
    model = RidgeCV(alphas=[100, 150, 200, 500, 600, 700], scoring='neg_mean_squared_error')
    model.fit(trainX, trainY)

    print('best alpha:', model.alpha_)
    a = model.alpha_
    model = RidgeCV(alphas=[a * .5, a * .55, a * .6, a * .65, a * .7, a * .75, a * .8,
                            a * .85, a * .9, a * .95, a * 1, a * 1.05, a * 1.1, a * 1.15,
                            a * 1.2, a * 1.25, a * 1.3, a * 1.35, a * 1.4], scoring='neg_mean_squared_error',
                            store_cv_values=True)
    model.fit(trainX, trainY)

    print('best alpha:', model.alpha_)
    print('picked ' + str(sum(model.coef_ != 0)) + ' features and eliminated the other ' + str(sum(model.coef_ == 0)) + ' features')
    print('score = %f'%model.score(dataX, dataY))

    print('=' * 10 + 'RidgeCV' + '=' * 10)
    return model

def GetLassoCV(dataX, dataY):
    print('=' * 10 + 'LassoCV' + '=' * 10)
    alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]
    model = LassoCV(alphas = alphas, max_iter=1000, cv=10)
    model.fit(dataX, dataY)

    print('best alpha=%f converged at iter=%d'%(model.alpha_, model.n_iter_))
    a = model.alpha_
    model = LassoCV(alphas=[a * .5, a * .55, a * .6, a * .65, a * .7, a * .75, a * .8,
                            a * .85, a * .9, a * .95, a * 1, a * 1.05, a * 1.1, a * 1.15,
                            a * 1.2, a * 1.25, a * 1.3, a * 1.35, a * 1.4], max_iter=1000, cv=10)
    model.fit(trainX, trainY)
    print('best alpha=%f converged at iter=%d'%(model.alpha_, model.n_iter_))
    print('picked ' + str(sum(model.coef_ != 0)) + ' features and eliminated the other ' + str(sum(model.coef_ == 0)) + ' features')
    print('score = %f'%model.score(dataX, dataY))
    
    print('=' * 10 + 'LassoCV' + '=' * 10)
    return model

def GetElasticNet(dataX, dataY):
    print('=' * 10 + 'ElasticNet' + '=' * 10)
    print('=' * 20)
    l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
    alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]
    model = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, max_iter=1000)
    model.fit(dataX, dataY)
    a = model.alpha_
    l1 = model.l1_ratio_
    print('best alpha=%f l1=%f converged at iter=%d'%(model.alpha_, model.l1_ratio_, model.n_iter_))
    print('=' * 20)
    print('Try again for more precision with l1_ratio centered around %f'%l1)
    model = ElasticNetCV(alphas=alphas, max_iter=1000, 
                       l1_ratio=[l1 * .5, l1 * .55, l1 * .6, l1 * .65, l1 * .7, l1 * .75, l1 * .8,
                                 l1 * .85, l1 * .9, l1 * .95, l1 * 1, l1 * 1.05, l1 * 1.1, l1 * 1.15,
                                 l1 * 1.2, l1 * 1.25, l1 * 1.3, l1 * 1.35, l1 * 1.4])
    model.fit(dataX, dataY)
    a = model.alpha_
    l1 = model.l1_ratio_
    print('best alpha=%f l1=%f converged at iter=%d'%(model.alpha_, model.l1_ratio_, model.n_iter_))
    print('=' * 20)
    print('Try again for more precision with alpha centered around %f'%a)
    model = ElasticNetCV(alphas=[a * .5, a * .55, a * .6, a * .65, a * .7, a * .75, a * .8,
                            a * .85, a * .9, a * .95, a * 1, a * 1.05, a * 1.1, a * 1.15,
                            a * 1.2, a * 1.25, a * 1.3, a * 1.35, a * 1.4], max_iter=1000,
                       l1_ratio=l1)
    model.fit(dataX, dataY)
    print('best alpha=%f l1=%f converged at iter=%d'%(model.alpha_, model.l1_ratio_, model.n_iter_))
    print('picked ' + str(sum(model.coef_ != 0)) + ' features and eliminated the other ' + str(sum(model.coef_ == 0)) + ' features')
    print('score = %f'%model.score(dataX, dataY))
    
    print('=' * 10 + 'ElasticNet' + '=' * 10)
    return model

def cross_validation(model, tX, tY, vX, vY):
    tScores = cross_val_score(model, tX, tY, scoring='neg_mean_squared_error', cv=10)
    tRMSE = np.sqrt(-tScores).mean()

    vScores = cross_val_score(model, vX, vY, scoring='neg_mean_squared_error', cv=10)
    vRMSE = np.sqrt(-vScores).mean()

    print('RMSE on Training set:', tRMSE)
    print('RMSE on Validation set:', vRMSE)
    
    tPred = model.predict(tX)
    vPred = model.predict(vX)

    #plot residuals
    plt.scatter(tPred, tPred - tY, c='blue', marker='s', label='Training data')
    plt.scatter(vPred, vPred - vY, c='red', marker='s', label='Validation data')
    plt.title('Cross validation')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    xmin = np.min([tPred.min(), vPred.min()])
    xmax = np.max([tPred.max(), vPred.max()])
    plt.hlines(y=0, xmin=xmin, xmax=xmax, color='black')
    plt.show()

    #plot predictions
    plt.scatter(tPred, tY, c='blue', marker='s', label='Training data')
    plt.scatter(vPred, vY, c='red', marker='s', label='Validation data')
    plt.title('Cross validation')
    plt.xlabel('Predicted values')
    plt.ylabel('Real values')
    plt.legend(loc='upper left')
    xmin = np.min([tPred.min(), vPred.min()])
    xmax = np.max([tPred.max(), vPred.max()])
    ymin = np.min([tY.min(), vY.min()])
    ymax = np.max([tY.max(), vY.max()])
    plt.plot([xmin, xmax], [ymin, ymax], c='black')
    plt.show()

    #plot important coefficients
    coefs = pd.Series(model.coef_, index=tX.columns)
    print('Model picked ' + str(sum(coefs != 0)) + 'features and eliminated the other ' + str(sum(coefs == 0)) + ' features')
    imp_coefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
    imp_coefs.plot(kind='barh')
    plt.title('Coefficients in the model')
    plt.show()

#%%
train = pd.read_csv('house-pricing/data/train.csv')
#convert categorical
train = pd.get_dummies(train)
#impute
imputer = Imputer(strategy='median')
train.loc[:, :] = imputer.fit_transform(train.loc[:, :])

trainX = train.drop(['SalePrice', 'Id'], axis=1)
trainY = train['SalePrice']

#log transfrom of the skewed features
skewness = trainX.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewnessF = skewness.index
trainX[skewnessF] = np.log1p(trainX[skewnessF])
trainY = np.log1p(trainY)

stdScaler = StandardScaler()
trainX.loc[:, :] = stdScaler.fit_transform(trainX.loc[:, :])

# tX, vX, tY, vY = train_test_split(trainX, trainY, test_size=0.3, random_state=0)
# cross_validation(model, tX, tY, vX, vY)
model = GetXGBRegressor(trainX, trainY)

test = pd.read_csv('house-pricing/data/test.csv')
testId = test['Id']
test = test.drop('Id', axis=1)
test = pd.get_dummies(test)
test.loc[:, :] = imputer.fit_transform(test.loc[:, :])

missF = None
if trainX.columns.size > test.columns.size:
    temp = trainX.drop(test.columns, axis=1)
    missF = temp.columns

for f in missF:
    test[f] = 0

skewness = test.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewnessF = skewness.index
test[skewnessF] = np.log1p(test[skewnessF])
test.loc[:, :] = stdScaler.fit_transform(test.loc[:, :])

if type(model) == XGBRegressor:
    test = test.as_matrix()

pred = model.predict(test)
result = pd.DataFrame({
    'Id': testId,
    'SalePrice': np.exp(pred) - 1
})

result.to_csv('house-pricing/submission/result_' + model.__class__.__name__ + '.csv', index=False)
print('Done')