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

def selectBestModelAndTrain(data, dataY):
    corr = data.select_dtypes(include = ['float64','int64']).corr()
    #找出和SalePrice相关性高的特征
    sortedSalePriceRelative = corr['SalePrice'].drop('SalePrice').sort_values(ascending=False)
    sspr = sortedSalePriceRelative
 
    highRelative = sspr[sspr > 0.5]
    #OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,1stFlrSF,FullBath
    #TotRmsAbvGrd,YearBuilt,YearRemodAdd
    #hrf = high relative features
    hrf = highRelative.axes[0].tolist()
    log_transform(data, hrf)
    dataY = log_transform(dataY, None)
    
    #处理定性参数
    mapping = {}
    objFeatures = data.select_dtypes(include=['object'])
    cols = objFeatures.columns
    objf = []
    for feature in cols:
        newCol = encode(data, feature, mapping)
        objf.append(newCol)
    
    #多模型交叉验证
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

    nullrows = data.isnull().any(axis=1)
    data = data[nullrows == False]
    dataY = dataY[nullrows == False]
    
    for modelName in models:
        model = models[modelName]
        scores = cross_val_score(model, data[hrf + objf], dataY, cv=3, scoring='neg_mean_squared_error')
        score = np.sqrt(-scores.mean())
        print('%s rmse scores = %0.3f'%(modelName, score))
        if score < bestScore:
            bestScore = score
            bestModel = model
    
    bestModel.fit(data[hrf + objf], dataY)
    return (bestModel, mapping, hrf, objf)

def dropNAFeatures(data, features):
    data.drop(features, inplace=True, axis=1)

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
    
    return data

def encode(data, feature, mapping):
    ordering = pd.DataFrame()
    ordering['val'] = data[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = data[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['ordering'].to_dict()
    mapping[feature] = ordering
    for cat, o in ordering.items():
        data.loc[data[feature] == cat, feature + '_E'] = o

    return feature + '_E'


naFeatures = ['Id', 'Alley', 'LotFrontage', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
dropNAFeatures(dtTrain, naFeatures)

model, mapping, hrf, objf = selectBestModelAndTrain(dtTrain, dtTrainY)

#%%
dtTestX = pd.read_csv('house-pricing/data/test.csv')
dtTestId = dtTestX['Id']
dropNAFeatures(dtTestX, naFeatures)

dtTestX['GarageCars'].fillna(dtTestX['GarageCars'].mean(), inplace=True)
dtTestX['GarageArea'].fillna(dtTestX['GarageArea'].mean(), inplace=True)
dtTestX['TotalBsmtSF'].fillna(dtTestX['TotalBsmtSF'].mean(), inplace=True)

#转换定性特征
for feature in objf:
    orif = feature[:-2]
    m = mapping[orif]
    for key, value in m.items():
        dtTestX.loc[dtTestX[orif] == key, feature] = value

dtTestX = dtTestX.fillna(0.0)
dtTestX[objf].isnull().sum()
log_transform(dtTestX, hrf)

pred = model.predict(dtTestX[hrf + objf])
result = pd.DataFrame({ 
  'Id': dtTestId,
  'SalePrice': np.exp(pred) - 1
})

result.to_csv('house-pricing/submission/hrf_objf_log_transform.csv', index=False)