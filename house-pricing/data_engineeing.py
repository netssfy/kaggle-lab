'data engineering'
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_val_score

dtTrain = pd.read_csv('house-pricing/data/train.csv')
dtTrainY = dtTrain['SalePrice']
dtTrainX = dtTrain.drop(['SalePrice'], axis=1)

dtTestX = pd.read_csv('house-pricing/data/test.csv')
dtTestId = dtTestX['Id']
#check NA nums
#dtTrain.info()
#Alley 107
#LotFrontage 1232
#FireplaceQu 729
#PoolQC 3
#Fence 290
#MiscFeature 51
#id也要移除,所以放这里
naFeatures = ['Id', 'Alley', 'LotFrontage', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
def dropNAFeatures(data, features):
  data.drop(features, inplace=True, axis=1)

dropNAFeatures(dtTrain, naFeatures)
dropNAFeatures(dtTrainX, naFeatures)
dropNAFeatures(dtTestX, naFeatures)

#画直方图
sns.distplot(dtTrainY, kde=False, color='b', hist_kws={ 'alpha': 0.9 })

#画关系图, 移除Id列
corr = dtTrain.select_dtypes(include = ['float64','int64']).corr()
sns.heatmap(corr, vmax=1, square=True)

#找出和SalePrice
sortedSalePriceRelative = corr['SalePrice'].drop('SalePrice').sort_values(ascending=False)
sspr = sortedSalePriceRelative
highRelative = sspr[sspr > 0.5]
#OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,1stFlrSF,FullBath
#TotRmsAbvGrd,YearBuilt,YearRemodAdd
#hrf = high relative features
hrf = highRelative.axes[0].tolist()

def selectHRFeatures(data, features):
  return data[features]

dtTrain = selectHRFeatures(dtTrain, hrf)
dtTrainX = selectHRFeatures(dtTrainX, hrf)
dtTestX = selectHRFeatures(dtTestX, hrf)

#更新后的热图
corr = dtTrain.corr()
sns.heatmap(corr, vmax=1, square=True)

#交叉验证
#%%
scores = cross_val_score(LR(), dtTrainX, dtTrainY, cv=3, scoring='mean_squared_error')

#%%
dtTestX['GarageCars'].fillna(dtTestX['GarageCars'].mean(), inplace=True)
dtTestX['GarageArea'].fillna(dtTestX['GarageArea'].mean(), inplace=True)
dtTestX['TotalBsmtSF'].fillna(dtTestX['TotalBsmtSF'].mean(), inplace=True)

model = LR()
model.fit(dtTrainX, dtTrainY)
pred = model.predict(dtTestX)
result = pd.DataFrame({ 
  'Id': dtTestId,
  'SalePrice': pred
})
result.to_csv('house-pricing/submission/hrf.csv', index=False)