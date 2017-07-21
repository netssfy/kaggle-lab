'playground'

#%%

import numpy as np
import pandas as pd

data = pd.read_csv('house-pricing/data/train.csv')

#%%
type(data['MSZoning'].mode())
