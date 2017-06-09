'data engineering'
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.linear_model import RidgeCV
from scipy.stats import skew

pd.set_option('display.float_format', lambda x: '%.6f'%x)

def cross_validation(model, tX, tY, vX, vY):
    tScores = cross_val_score(model, tX, tY, scoring='neg_mean_squared_error', cv=10)
    tRMSE = np.sqrt(-tScores).mean()

    vScores = cross_val_score(model, vX, vY, scoring='neg_mean_squared_error', cv=10)
    vRMSE = np.sqrt(-vScores).mean()

    print('RMSE on Training set:', tRMSE)
    print('RMSE on Validation set:', vRMSE)
    
    tPred = model.predict(tX)
    vPred = model.predict(vX)

    #plot residuals
    plt.scatter(tPred, tPred - tY, c='blue', marker='s', label='Training data')
    plt.scatter(vPred, vPred - vY, c='red', marker='s', label='Validation data')
    plt.title('Cross validation')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    xmin = np.min([tPred.min(), vPred.min()])
    xmax = np.max([tPred.max(), vPred.max()])
    plt.hlines(y=0, xmin=xmin, xmax=xmax, color='black')
    plt.show()

    #plot predictions
    plt.scatter(tPred, tY, c='blue', marker='s', label='Training data')
    plt.scatter(vPred, vY, c='red', marker='s', label='Validation data')
    plt.title('Cross validation')
    plt.xlabel('Predicted values')
    plt.ylabel('Real values')
    plt.legend(loc='upper left')
    xmin = np.min([tPred.min(), vPred.min()])
    xmax = np.max([tPred.max(), vPred.max()])
    ymin = np.min([tY.min(), vY.min()])
    ymax = np.max([tY.max(), vY.max()])
    plt.plot([xmin, xmax], [ymin, ymax], c='black')
    plt.show()

    #plot important coefficients
    coefs = pd.Series(model.coef_, index=tX.columns)
    print('Model picked ' + str(sum(coefs != 0)) + 'features and eliminated the other ' + str(sum(coefs == 0)) + ' features')
    imp_coefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
    imp_coefs.plot(kind='barh')
    plt.title('Coefficients in the model')
    plt.show()

#%%
train = pd.read_csv('house-pricing/data/train.csv')
#convert categorical
train = pd.get_dummies(train)
#impute
imputer = Imputer(strategy='median')
train.loc[:, :] = imputer.fit_transform(train.loc[:, :])

trainX = train.drop('SalePrice', axis=1)
trainY = train['SalePrice']

#log transfrom of the skewed features
skewness = trainX.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewnessF = skewness.index
trainX[skewnessF] = np.log1p(trainX[skewnessF])
trainY = np.log1p(trainY)

stdScaler = StandardScaler()
trainX.loc[:, :] = stdScaler.fit_transform(trainX.loc[:, :])
trainY.loc[:] = stdScaler.fit_transform(trainY.loc[:])

tX, vX, tY, vY = train_test_split(trainX, trainY, test_size=0.3, random_state=0)

model = RidgeCV(alphas=[100, 150, 200, 500, 600, 700], scoring='neg_mean_squared_error')
model.fit(trainX, trainY)

print(model.alpha_)
a = model.alpha_

model = RidgeCV(alphas=[a * .5, a * .55, a * .6, a * .65, a * .7, a * .75, a * .8,
                        a * .85, a * .9, a * .95, a * 1, a * 1.05, a * 1.1, a * 1.15,
                        a * 1.2, a * 1.25, a * 1.3, a * 1.35, a * 1.4], scoring='neg_mean_squared_error')
model.fit(trainX, trainY)

print(model.alpha_)
# cross_validation(model, tX, tY, vX, vY)

test = pd.read_csv('house-pricing/data/test.csv')
test = pd.get_dummies(test)
test.loc[:, :] = imputer.fit_transform(test.loc[:, :])

print(test.isnull().sum())
test1, _ = test.align(trainX)
print(trainX.shape)
print(test1.shape)
print(test1.isnull().sum())

# skewness = test.apply(lambda x: skew(x))
# skewness = skewness[abs(skewness) > 0.5]
# skewnessF = skewness.index
# test[skewnessF] = np.log1p(test[skewnessF])
# test.loc[:, :] = stdScaler.fit_transform(test.loc[:, :])

# pred = model.predict(test)
# pred