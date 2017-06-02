'data engineering'
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.linear_model import Lasso as Lasso
from sklearn.linear_model import ElasticNet as EN
from sklearn.linear_model import Ridge as RR
from sklearn.svm import LinearSVR as SVR
from sklearn.ensemble import RandomForestRegressor as RFR
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

#多模型交叉验证
#%%
models = {
  'Linear Regression': LR(),
  'SGD Regressor': SGD(),
  'Lasso': Lasso(),
  'Elastic': EN(),
  'Ridge Regression': RR(),
  'Linear SVR': SVR(),
  'Random Forest Regressor': RFR()
}

bestScore = 1000000000000
bestModel = None

for modelName in models:
  model = models[modelName]
  scores = cross_val_score(model, dtTrainX, dtTrainY, cv=3, scoring='neg_mean_squared_error')
  score = np.sqrt(-scores.mean())
  print('%s rmse scores = %0.3f'%(modelName, score))
  if score < bestScore:
    bestScore = score
    bestModel = model

#%%
dtTestX['GarageCars'].fillna(dtTestX['GarageCars'].mean(), inplace=True)
dtTestX['GarageArea'].fillna(dtTestX['GarageArea'].mean(), inplace=True)
dtTestX['TotalBsmtSF'].fillna(dtTestX['TotalBsmtSF'].mean(), inplace=True)

print(type(bestModel))
bestModel.fit(dtTrainX, dtTrainY)
pred = bestModel.predict(dtTestX)
result = pd.DataFrame({ 
  'Id': dtTestId,
  'SalePrice': pred
})
result.to_csv('house-pricing/submission/hrf.csv', index=False)