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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from scipy.special import boxcox1p
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error

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

#将某些数值型的特征转换成类目型
mx['MSSubClass'] = mx['MSSubClass'].apply(str)
mx['OverallCond'] = mx['OverallCond'].apply(str)
mx['YrSold'] = mx['YrSold'].apply(str)
mx['MoSold'] = mx['MoSold'].apply(str)

#离散变量编码
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    mx[c] = lbl.fit_transform(mx[c].values)


#增加特征
mx['TotalSF'] = mx['TotalBsmtSF'] + mx['1stFlrSF'] + mx['2ndFlrSF']
# mx['TotalBath'] = mx['FullBath'] + mx['HalfBath']

# mx['BathPerBedroom'] = mx['TotalBath'] / mx['BedroomAbvGr']
# mx['BathPerBedroom'] = mx['BathPerBedroom'].replace([np.inf, -np.inf], 0)
# mx['BathPerBedroom'] = mx['BathPerBedroom'].fillna(0)

# mx['KitchenPerBedroom'] = mx['KitchenAbvGr'] / mx['BedroomAbvGr']
# mx['KitchenPerBedroom'] = mx['KitchenPerBedroom'].replace([np.inf, -np.inf], 0)
# mx['KitchenPerBedroom'] = mx['KitchenPerBedroom'].fillna(0)

# mx['KitchenPerBath'] = mx['KitchenAbvGr'] / mx['TotalBath']
# mx['KitchenPerBath'] = mx['KitchenPerBath'].replace([np.inf, -np.inf], 0)
# mx['KitchenPerBath'] = mx['KitchenPerBath'].fillna(0)

# mx['GarageCarsPerBedroom'] = mx['GarageCars'] / mx['BedroomAbvGr']
# mx['GarageCarsPerBedroom'] = mx['GarageCarsPerBedroom'].replace([np.inf, -np.inf], 0)
# mx['GarageCarsPerBedroom'] = mx['GarageCarsPerBedroom'].fillna(0)

#分类特征dummy
mx = pd.get_dummies(mx)

#对倾斜大的特征做log
skewness = mx.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.75]
skewnessF = skewness.index
lam = 0.15
for feat in skewnessF:
    mx[feat] = boxcox1p(mx[feat], lam)
    mx[feat] += 1

#对y做log,使其更正态
trainY = np.log1p(rawTrainY)

n_folds = 5

def rmse_cv(model, dataX, dataY):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataX)
    rmse = np.sqrt(-cross_val_score(model, dataX, dataY, scoring='neg_mean_squared_error', cv=kf))
    return rmse

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# stacking models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for train, holdout in kfold.split(X, y):
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train], y.iloc[train])
                y_pred = instance.predict(X.iloc[holdout])
                out_of_fold_predictions[holdout, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def GetXGBRegressor(X, Y):
    X = X.as_matrix()

    print('=' * 10 + 'XGBRegressor' + '=' * 10)
    # etas = [0.0925, 0.093, 0.0935]
    # depths = [10, 12]
    # child_weights = [3, 4, 5]
    # gammas = [0.0341, 0.0342, 0.0343]
    # params = []
    # for a1 in etas:
    #     for a2 in depths:
    #         for a3 in child_weights:
    #             for a4 in gammas:
    #                 params.append({ 'eta': a1, 'depth': a2, 'child_weight': a3, 'gamma': a4})

    # bestScore = 10000000
    # bestParam = None
    # bestModel = None
    # for param in params:
    #     print('running ' + str(param))
    #     model = XGBRegressor(learning_rate=param['eta'], max_depth=param['depth'], min_child_weight=param['child_weight'], gamma=param['gamma'])
    #     scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
    #     score = np.sqrt(-scores).mean()

    #     print('score=%f'%(score))
    #     if score < bestScore:
    #         bestScore = score
    #         bestParam = param
    #         bestModel = model
    
    bestParam = {'child_weight': 4, 'depth': 10, 'eta': 0.093, 'gamma': 0.0341}
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


length = rawTrainX.shape[0]
trainX = mx[:length]
testX = mx[length:]

# model selecting
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1)

stacked_averaged_models = StackingAveragedModels(base_models = [ENet, GBoost, KRR],
                                                 meta_model = lasso)

model_rf = RandomForestRegressor()
#%%
#ensembling model
#StackedRegressor
stacked_averaged_models.fit(trainX, trainY)
stacked_train_pred = stacked_averaged_models.predict(trainX)
stacked_pred = np.expm1(stacked_averaged_models.predict(testX))
print('stacked rmse = {}'.format(rmse(trainY, stacked_train_pred)))

#XGBoost
model_xgb.fit(trainX, trainY)
xgb_train_pred = model_xgb.predict(trainX)
xgb_pred = np.expm1(model_xgb.predict(testX))
print('xgb rmse = {}'.format(rmse(trainY, xgb_train_pred)))

#random forrest
model_rf.fit(trainX, trainY)
rf_train_pred = model_rf.predict(trainX)
rf_pred = np.expm1(model_rf.predict(testX))

print('random forrest = {}'.format(rmse(trainY, rf_train_pred)))
'''RMSE on the entire Train data when averaging'''
print('RMSE score on train data')

#%%
A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
B = A
bestRmse = 100
bestParam = None

for a in A:
    for b in B:
        if a + b >= 1:
            continue
        c = 1 - a - b
        val = rmse(trainY, stacked_train_pred * a + xgb_train_pred * b + rf_train_pred * c)
        print('param a={} b={} c={} rmse={}'.format(a, b, c, val))
        if val < bestRmse:
            bestRmse = val
            bestParam = (a, b, c)

print('best a={} b={} c={} rmse={}'.format(bestParam[0], bestParam[1], bestParam[2], bestRmse))

ensemble = stacked_pred * bestParam[0] + xgb_pred * bestParam[1] + rf_pred * bestParam[2]

result = pd.DataFrame({
    'Id': testId,
    'SalePrice': ensemble
})

result.to_csv('house-pricing/submission/result3_ensemble.csv', index=False)
print('Done')

# model = GetXGBRegressor(trainX, trainY)
# 'child_weight': 4, 'depth': 10, 'eta': 0.093, 'gamma': 0.0341
# model = XGBRegressor(
#                     colsample_bytree=0.2, gamma=0.0, 
#                     learning_rate=0.05, max_depth=6, 
#                     min_child_weight=1.5, n_estimators=7200,
#                     reg_alpha=0.9, reg_lambda=0.6,
#                     subsample=0.2, seed=42, silent=1, random_state =7,
#                     child_weight=4, depth=10, eta=0.093
#                     )

# if type(model) == XGBRegressor:
#     testX = testX.as_matrix()

# pred = model.predict(testX)
# result = pd.DataFrame({
#     'Id': testId,
#     'SalePrice': np.exp(pred) - 1
# })

# result.to_csv('house-pricing/submission/result3_' + model.__class__.__name__ + '.csv', index=False)
# print('Done')