'data engineering'
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dtTrain = pd.read_csv('house-pricing/data/train.csv')
#check NA nums
#dtTrain.info()
#Alley 107
#LotFrontage 1232
#FireplaceQu 729
#PoolQC 3
#Fence 290
#MiscFeature 51

exclude = ['Id', 'SalePrice']
colNames = dtTrain.keys()
for name in colNames:
  if name not in exclude:
    dtTrain.plot(name, 'SalePrice', kind='scatter')
#dtTrain.plot('LotFrontage', 'SalePrice', kind='scatter')

#sns.lmplot('LotFrontage', 'SalePrice', dtTrain)