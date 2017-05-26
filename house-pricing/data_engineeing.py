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
#check NA nums
#dtTrain.info()
#Alley 107
#LotFrontage 1232
#FireplaceQu 729
#PoolQC 3
#Fence 290
#MiscFeature 51

# exclude = ['Id', 'SalePrice']
# colNames = dtTrain.keys()
# for name in colNames:
#   if name not in exclude:
#     dtTrain.plot(name, 'SalePrice', kind='scatter')
#dtTrain.plot('LotFrontage', 'SalePrice', kind='scatter')

#sns.lmplot('LotFrontage', 'SalePrice', dtTrain)

#dtTrainX.shape
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