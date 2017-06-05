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

def plot(dataX, dataY):
    #画直方图
    sns.distplot(dataY, kde=False, color='b', hist_kws={ 'alpha': 0.9 })
    #画关系图, 移除Id列
    corr = dataX.select_dtypes(include = ['float64','int64']).corr()
    sns.heatmap(corr, vmax=1, square=True)

def log_transform(data, features):
    if features is not None:
        data[features] = np.log1p(data[features].values)
    else:
        data = np.log1p(data)

def processQuantitativeFeatures(data):
    corr = data.select_dtypes(include = ['float64','int64']).corr()
    #找出和SalePrice相关性高的特征
    sortedSalePriceRelative = corr['SalePrice'].drop('SalePrice').sort_values(ascending=False)
    sspr = sortedSalePriceRelative
    highRelative = sspr[sspr > 0.5]
    #OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,1stFlrSF,FullBath
    #TotRmsAbvGrd,YearBuilt,YearRemodAdd
    #hrf = high relative features
    hrf = highRelative.axes[0].tolist()
    quanTrainX = selectFeatures(dtTrainX, hrf)
    log_transform(quanTrainX, hrf)
    return (quanTrainX, hrf)

def processQualitativeFeatures(data):
    objFeatures = data.select_dtypes(include=['object'])
    return objFeatures

def selectFeatures(data, features):
    return data[features]

quanHRF, hrf = processQuantitativeFeatures(dtTrain)

#%%
#定性参数
processQualitativeFeatures(dtTrainX).columns

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
    scores = cross_val_score(model, quanHRF, dtTrainY, cv=3, scoring='neg_mean_squared_error')
    score = np.sqrt(-scores.mean())
    print('%s rmse scores = %0.3f'%(modelName, score))
    if score < bestScore:
        bestScore = score
        bestModel = model

#%%
dtTestX['GarageCars'].fillna(dtTestX['GarageCars'].mean(), inplace=True)
dtTestX['GarageArea'].fillna(dtTestX['GarageArea'].mean(), inplace=True)
dtTestX['TotalBsmtSF'].fillna(dtTestX['TotalBsmtSF'].mean(), inplace=True)

quanTestX = selectFeatures(dtTestX, hrf)
log_transform(quanTestX, hrf)

print(type(bestModel))
bestModel.fit(quanHRF, dtTrainY)
pred = bestModel.predict(quanTestX)
result = pd.DataFrame({ 
  'Id': dtTestId,
  'SalePrice': np.exp(pred) - 1
})
result.to_csv('house-pricing/submission/hrf_log_transform.csv', index=False)

#=================================================================
#=================================================================
#%%
#object类型特征
dtTrain = pd.read_csv('house-pricing/data/train.csv')
dtTrainY = dtTrain['SalePrice']
dtTrainX = dtTrain.drop('SalePrice', axis=1)

qualitative = [f for f in dtTrain.columns if dtTrain.dtypes[f] == 'object']
qualitative
#%%
def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
for q in qualitative:  
    encode(dtTrain, q)
    qual_encoded.append(q+'_E')

dtTrain[qual_encoded]
# plt.figure(figsize = (12, 6))
# sns.boxplot(x='Neighborhood', y='SalePrice', data=dtTrain)
# plt.xticks(rotation=45)
# plt.figure(figsize = (12, 6))
# sns.countplot(x='Neighborhood', data=dtTrain)
# plt.xticks(rotation=45)