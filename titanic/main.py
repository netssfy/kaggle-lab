'''hello world'''

#%%
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt

trainData = pd.read_csv('./titanic/train.csv')
testData = pd.read_csv('./titanic/test.csv')

Y = trainData['Survived']
trainData = trainData.drop('Survived', axis=1)
X = pd.concat([trainData, testData])

ID = X['PassengerId']
X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X['Sex'] = X['Sex'].replace('male', 1)
X['Sex'] = X['Sex'].replace('female', 0)

X['Family'] = X['SibSp'] + X['Parch']

X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
X['Embarked'] = X['Embarked'].replace('C', 0)
X['Embarked'] = X['Embarked'].replace('Q', 1)
X['Embarked'] = X['Embarked'].replace('S', 2)

X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Age'] = X['Age'].fillna(X['Age'].median())

X.info()