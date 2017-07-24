'playground'

#%%

import numpy as np
import pandas as pd

all_data = pd.read_csv('house-pricing/data/train.csv')
print all_data['FireplaceQu'][0:5]
print all_data['BsmtQual'][0:5]
#%%
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    all_data[c] = lbl.fit_transform(all_data[c].values)

print all_data['FireplaceQu'][0:5]
print all_data['BsmtQual'][0:5]