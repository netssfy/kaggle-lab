'data engineering'

#%%
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sns
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import Imputer

rawTrain = pd.read_csv('house-pricing/data/train.csv')
rawTrainX = rawTrain.drop(['SalePrice', 'Id'], axis=1)
rawTrainY = rawTrain['SalePrice']

rawTest = pd.read_csv('house-pricing/data/test.csv')
rawTestX = rawTest.drop(['Id'], axis=1)

#讲两者合并,再处理,因为有些分类特征可能在test里有,但是在train里没有
mergedX = pd.concat([rawTrainX, rawTestX])

print('raw train info = ' + str(rawTrainX.shape))
print('raw test info = ' + str(rawTestX.shape))
print('merged train info = ' + str(mergedX.shape))

mergedX = mergedX.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
mergedX.info()
#%%
#增加特征
mergedX['TotalBath'] = mergedX['FullBath'] + mergedX['HalfBath']
mergedX['BathPerBedroom'] = mergedX['TotalBath'] / mergedX['BedroomAbvGr']
#mergedX['BathPerRoom'].fillna(0)

mergedX['KitchenPerBedroom'] = mergedX['KitchenAbvGr'] / mergedX['BedroomAbvGr']
mergedX['KitchenPerBath'] = mergedX['KitchenAbvGr'] / mergedX['TotalBath']

mergedX['GarageCarsPerBedroom'] = mergedX['GarageCars'] / mergedX['BedroomAbvGr']

#对y做log,使其更正态
trainY = np.log1p(rawTrainY)

#分类特征dummy
mergedX = pd.get_dummies(mergedX)

#对NA做处理
imputer = Imputer(strategy='median')
mergedX.loc[:, :] = imputer.fit_transform(mergedX.loc[:, :])

#对倾斜大的特征做log
skewness = mergedX.apply(lambda x: skew(x))