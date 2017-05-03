#%%
import numpy as np
import pandas as pd

train_dt = pd.read_csv('titanic/train.csv')
test_dt = pd.read_csv('titanic/test.csv')

train_dt.info()

#0 female 1 male 2 child
def define_person(info):
    age, sex = info
    return 2 if age < 16 else (0 if sex == 'female' else 1)

def process_data(data):
    data['Person'] = data[['Age', 'Sex']].apply(define_person, axis=1)
    data.drop(['Sex'], axis=1)

process_data(train_dt)
train_dt['Person']
