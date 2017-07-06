'data engineering'

#%%
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sns
import pandas as pd
# import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost.sklearn import XGBRegressor

rawTrain = pd.read_csv('house-pricing/data/train.csv')
rawTrainX = rawTrain.drop(['SalePrice', 'Id'], axis=1)
rawTrainY = rawTrain['SalePrice']

rawTest = pd.read_csv('house-pricing/data/test.csv')
testId = rawTest['Id']
rawTestX = rawTest.drop(['Id'], axis=1)

#讲两者合并,再处理,因为有些分类特征可能在test里有,但是在train里没有
mergedX = pd.concat([rawTrainX, rawTestX])

print('raw train info = ' + str(rawTrainX.shape))
print('raw test info = ' + str(rawTestX.shape))
print('merged train info = ' + str(mergedX.shape))

# mergedX = mergedX.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

#增加特征
mergedX['TotalBath'] = mergedX['FullBath'] + mergedX['HalfBath']
mergedX['BathPerBedroom'] = mergedX['TotalBath'] / mergedX['BedroomAbvGr']
#mergedX['BathPerRoom'].fillna(0)

mergedX['KitchenPerBedroom'] = mergedX['KitchenAbvGr'] / mergedX['BedroomAbvGr']
mergedX['KitchenPerBath'] = mergedX['KitchenAbvGr'] / mergedX['TotalBath']

mergedX['GarageCarsPerBedroom'] = mergedX['GarageCars'] / mergedX['BedroomAbvGr']

mergedX = mergedX.replace([np.inf, -np.inf], 0.0001)

#对y做log,使其更正态
trainY = np.log1p(rawTrainY)

#分类特征dummy
mergedX = pd.get_dummies(mergedX)

#对NA做处理
imputer = Imputer(strategy='median')
mergedX.loc[:, :] = imputer.fit_transform(mergedX.loc[:, :])

#对倾斜大的特征做log
skewness = mergedX.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewnessF = skewness.index
mergedX[skewnessF] = np.log1p(mergedX[skewnessF])

#做标准化归一化处理
# stdScaler = StandardScaler()
# mergedX.loc[: ,:] = stdScaler.fit_transform(mergedX.loc[:, :])

def GetXGBRegressor(X, Y):
    # print('=' * 10 + 'XGBRegressor' + '=' * 10)
    # etas = [0.13, 0.15, 0.18]
    # depths = [7, 10]
    # child_weights = [3, 5, 8]
    # gammas = [0.26, 0.03, 0.034]
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
    
    bestParam = {'child_weight': 8, 'depth': 7, 'eta': 0.13, 'gamma': 0.034}
    print('best param = ' + str(bestParam))
    model = XGBRegressor(learning_rate=bestParam['eta'], max_depth=bestParam['depth'], min_child_weight=bestParam['child_weight'], gamma=bestParam['gamma'])
    X = X.as_matrix()
    model.fit(X, Y)
    print('=' * 10 + 'XGBRegressor' + '=' * 10)
    return model

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

print(mergedX.shape)
len = rawTrainX.shape[0]
trainX = mergedX[:len]
testX = mergedX[len:]

model = GetXGBRegressor(trainX, trainY)

if type(model) == XGBRegressor:
    testX = testX.as_matrix()

pred = model.predict(testX)
result = pd.DataFrame({
    'Id': testId,
    'SalePrice': np.exp(pred) - 1
})

result.to_csv('house-pricing/submission/result3_' + model.__class__.__name__ + '.csv', index=False)
print('Done')