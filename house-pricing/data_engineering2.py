'data engineering'
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

pd.set_option('display.float_format', lambda x: '%.3f'%x)

def cross_validation(model, tX, tY, vX, vY):
    tScores = cross_val_score(model, tX, tY, scoring='neg_mean_squared_error', cv=10)
    tRMSE = np.sqrt(-tScores).mean()

    vScores = cross_val_score(model, vX, vY, scoring='neg_mean_squared_error', cv=10)
    vRMSE = np.sqrt(-vScores).mean()

    print('RMSE on Training set:', tRMSE)
    print('RMSE on Validation set:', vRMSE)
    
    model.fit(tX, tY)
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
trainX = train.select_dtypes(exclude=['object'])
numericalF = trainX.columns
trainX = trainX.fillna(0.0)
trainY = train['SalePrice']

stdScaler = StandardScaler()
trainX.loc[:, numericalF] = stdScaler.fit_transform(trainX.loc[:, numericalF])
trainY.loc[:] = stdScaler.fit_transform(trainY.loc[:])

#%%
tX, vX, tY, vY = train_test_split(trainX, trainY, test_size=0.3, random_state=0)
print('train x:%s'%str(tX.shape))
print('validation x:%s'%str(vX.shape))
print('train y:%s'%str(tY.shape))
print('validation y:%s'%str(vY.shape))

model = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60], cv=10)

cross_validation(model, tX, tY, vX, vY)