'playground'

#%%

import numpy as np
import pandas as pd

all_data = pd.read_csv('house-pricing/data/train.csv')
print all_data['FireplaceQu'][0:5]
print all_data['BsmtQual'][0:5]
#%%
X = pd.DataFrame([[1,2],[3,4]], columns=['a', 'b'])

s = pd.Series([0, 1])
X.iloc[[0,1]]