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

#%%
#画关系图, 移除Id列
corr = dtTrain.select_dtypes(include = ['float64','int64']).corr()
sns.heatmap(corr, vmax=1, square=True)

#%%
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

dtTestX

# exclude = ['Id', 'SalePrice']
# colNames = dtTrain.keys()
# for name in colNames:
#   if name not in exclude:
#     dtTrain.plot(name, 'SalePrice', kind='scatter')
#dtTrain.plot('LotFrontage', 'SalePrice', kind='scatter')

#sns.lmplot('LotFrontage', 'SalePrice', dtTrain)

#%%
#找出所有number类型
numberFields = []
for name in dtTrainX.keys():
  dtype = type(dtTrainX[name][0])
  if dtype == np.int64 or dtype == np.float64:
    numberFields.append(name)
    dtTrainX[name].fillna(dtTrainX[name].mean(), inplace=True)

# scores = cross_val_score(LR(), dtTrainX['MSSubClass'], dtTrainY, cv=3, scoring='f1')
# scores
model = LR()
model.fit(dtTrainX[numberFields], dtTrainY)
pred = model.predict(dtTrainX[numberFields])
pred.shape