'playground'

#%%

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

mx = pd.read_csv('house-pricing/data/train.csv')
print mx['FireplaceQu'][0:5]
print mx['BsmtQual'][0:5]
#%%
[list() for x in [1,2,3]]