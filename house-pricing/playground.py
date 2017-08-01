'playground'

#%%

import numpy as np
import pandas as pd

mx = pd.read_csv('house-pricing/data/train.csv')
print mx['FireplaceQu'][0:5]
print mx['BsmtQual'][0:5]
#%%
mx['TotalSF'] = mx['TotalBsmtSF'] + mx['1stFlrSF'] + mx['2ndFlrSF']
mx['TotalBath'] = mx['FullBath'] + mx['HalfBath']
mx['BathPerBedroom'] = mx['TotalBath'] / mx['BedroomAbvGr']

print('a={} b={}'.format(1,2))