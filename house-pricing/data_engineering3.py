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

#删除outliers离群数据,只删除train上的
rawTrain = rawTrain.drop(rawTrain[(rawTrain['GrLivArea'] > 4000) & (rawTrain['SalePrice'] <  300000)].index)

rawTrainX = rawTrain.drop(['SalePrice', 'Id'], axis=1)
rawTrainY = rawTrain['SalePrice']

rawTest = pd.read_csv('house-pricing/data/test.csv')
testId = rawTest['Id']
rawTestX = rawTest.drop(['Id'], axis=1)

#将两者合并,再处理,因为有些分类特征可能在test里有,但是在train里没有
mergedX = pd.concat([rawTrainX, rawTestX])
#快捷方式
mx = mergedX
print('merged train info = ' + str(mx.shape))

#处理特征的缺失值
mx['PoolQC'] = mx['PoolQC'].fillna('None')
mx['MiscFeature'] = mx['MiscFeature'].fillna('None')
mx['Alley'] = mx['Alley'].fillna('None')
mx['Fence'] = mx['Fence'].fillna('None')
mx['FireplaceQu'] = mx['FireplaceQu'].fillna('None')

#门前的距离取同区域邻居数据的中位数
mx['LotFrontage'] = mx.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    mx[col] = mx[col].fillna('None')

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    mx[col] = mx[col].fillna(0)

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    mx[col] = mx[col].fillna(0)

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    mx[col] = mx[col].fillna('None')

mx['MasVnrType'] = mx['MasVnrType'].fillna('None')
mx['MasVnrArea'] = mx['MasVnrArea'].fillna(0)

mx['MSZoning'] = mx['MSZoning'].fillna(mx['MSZoning'].mode()[0])

#For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA .
#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
mx = mx.drop(['Utilities'], axis=1)

mx['Functional'] = mx['Functional'].fillna('Typ')

mx['Electrical'] = mx['Electrical'].fillna(mx['Electrical'].mode()[0])

mx['KitchenQual'] = mx['KitchenQual'].fillna(mx['KitchenQual'].mode()[0])

mx['Exterior1st'] = mx['Exterior1st'].fillna(mx['Exterior1st'].mode()[0])
mx['Exterior2nd'] = mx['Exterior2nd'].fillna(mx['Exterior2nd'].mode()[0])

mx['SaleType'] = mx['SaleType'].fillna(mx['SaleType'].mode()[0])

mx['MSSubClass'] = mx['MSSubClass'].fillna("None")

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
stdScaler = StandardScaler()
mergedX.loc[: ,:] = stdScaler.fit_transform(mergedX.loc[:, :])

def GetXGBRegressor(X, Y):
    X = X.as_matrix()

    print('=' * 10 + 'XGBRegressor' + '=' * 10)
    etas = [0.0925, 0.093, 0.0935]
    depths = [10, 12]
    child_weights = [3, 4, 5]
    gammas = [0.0341, 0.0342, 0.0343]
    params = []
    for a1 in etas:
        for a2 in depths:
            for a3 in child_weights:
                for a4 in gammas:
                    params.append({ 'eta': a1, 'depth': a2, 'child_weight': a3, 'gamma': a4})

    bestScore = 10000000
    bestParam = None
    bestModel = None
    for param in params:
        print('running ' + str(param))
        model = XGBRegressor(learning_rate=param['eta'], max_depth=param['depth'], min_child_weight=param['child_weight'], gamma=param['gamma'])
        scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
        score = np.sqrt(-scores).mean()

        print('score=%f'%(score))
        if score < bestScore:
            bestScore = score
            bestParam = param
            bestModel = model
    
    # bestParam = {'child_weight': 4, 'depth': 10, 'eta': 0.093, 'gamma': 0.0341}
    print('best param = ' + str(bestParam))
    model = XGBRegressor(learning_rate=bestParam['eta'], max_depth=bestParam['depth'], min_child_weight=bestParam['child_weight'], gamma=bestParam['gamma'])
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