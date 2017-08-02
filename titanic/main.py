'''hello world'''

#%%
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

trainData = pd.read_csv('./titanic/train.csv')
testData = pd.read_csv('./titanic/test.csv')
length = trainData.shape[0]
trainId = trainData['PassengerId']
testId = testData['PassengerId']

trainY = trainData['Survived']
trainData = trainData.drop('Survived', axis=1)
X = pd.concat([trainData, testData])

X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

X['Family'] = X['SibSp'] + X['Parch']
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Age'] = X['Age'].fillna(X['Age'].median())

X = pd.get_dummies(X)

trainX = X[:length]
testX = X[length:]

trainX.info()
#%%
from sklearn.base import clone
class StackingClassifer:
    def __init__(self, base_models, meta_model, folds=5):
        self.base_models = base_models
        self.meta_models = meta_model
        self.folds = folds

    def fit(X, Y):
        #每个model都会有多个实例对应各自fold
        self.current_base_models = [list() for x in self.base_models]
        self.current_meta_model = clone(self.meta_model)
        #对每个base模型都要做fit
        for 

        return self
#%%
mdl_rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
mdl_rfc.fit(trainX, trainY)
mdl_rfc_pred = mdl_rfc.predict(testX)

print mdl_rfc.score(trainX, trainY)

result = pd.DataFrame({
    'PassengerId': testId,
    'Survived': mdl_rfc_pred
})

result.to_csv('./titanic/submission/result.csv', index=False)