'playground'

#%%

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

mx = pd.read_csv('house-pricing/data/train.csv')
print mx['FireplaceQu'][0:5]
print mx['BsmtQual'][0:5]
#%%
a = np.random.rand(20, 10)
a.mean(axis=1)
# a[a > 0.5] = 1
# a[a < 0.5] = 0
# a