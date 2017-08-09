'''hello world'''

#%%
import copy
import re
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

trainData = pd.read_csv('./titanic/train.csv')
testData = pd.read_csv('./titanic/test.csv')
length = trainData.shape[0]
trainId = trainData['PassengerId']
testId = testData['PassengerId']

trainY = trainData['Survived']
trainData = trainData.drop('Survived', axis=1)
X = pd.concat([trainData, testData])

X = X.drop([], axis=1)

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ''

X['Family'] = X['SibSp'] + X['Parch'] + 1
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Age'] = X['Age'].fillna(X['Age'].median())

X['Pclass'] = X['Pclass'].apply(str)

X['Age2'] = X['Age'] ** 2
X['SibSp2'] = X['SibSp'] ** 2
X['Family2'] = X['Family'] ** 2
X['Fare2'] = X['Fare'] ** 2

X['Age3'] = X['Age'] ** 3
X['SibSp3'] = X['SibSp'] ** 3
X['Family3'] = X['Family'] ** 3
X['Fare3'] = X['Fare'] ** 3

X['Title'] = X['Name'].apply(get_title)
X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

X['Title'] = X['Title'].replace('Mlle', 'Miss')
X['Title'] = X['Title'].replace('Ms', 'Miss')
X['Title'] = X['Title'].replace('Mme', 'Mrs')

X = X.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)

X = pd.get_dummies(X)

trainX = X[:length]
testX = X[length:]

trainX.head()

class StackingClassifer:
    def __init__(self, base_models, meta_model, folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.folds = folds

    def fit(self, X, Y):
        #每个model都会有多个实例对应各自fold
        self.current_base_models = [list() for x in self.base_models]
        self.current_meta_model = clone(self.meta_model)
        kfold = KFold(n_splits=self.folds, shuffle=True)
        inter_pred = np.zeros((X.shape[0], len(self.base_models)))
        #对每个base模型都要做fit
        for i, model in enumerate(self.base_models):
            #对每个model都要做nfold的cross validation
            #走完这个循环i列就是这个model的预测结果了
            for train, test in kfold.split(X, Y):
                instance = clone(model)
                self.current_base_models[i].append(instance)
                instance.fit(X.iloc[train], Y.iloc[train])
                pred = instance.predict(X.iloc[test])
                inter_pred[test, i] = pred

        #使用中间预测结果,训练meta
        self.current_meta_model.fit(inter_pred, Y)
        return self

    def predict(self, X):
        #对每个base model都做预测，产生中间结果
        rows = X.shape[0]
        inter_pred = np.zeros((rows, len(self.current_base_models)))
        for i, models in enumerate(self.current_base_models):
            #由于训练时用了nfold所以会有多个实例
            #每个实例都会完整的跑一次预测，将结果求均值，大于0.5则认为1
            preds = np.zeros((rows, self.folds))
            for j, model in enumerate(models):
                preds[:, j] = model.predict(X)
            
            preds = preds.mean(axis=1)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            inter_pred[:, i] = preds

        return self.current_meta_model.predict(inter_pred)

    def score(self, X, Y):
        pred_y = self.predict(X)
        return (pred_y == Y).sum() / (float)(Y.shape[0])

def engineering_params(Model, params_map, orthogonal=False):
    print '*' * 20
    folds = KFold(n_splits=3)
    best_param = {}
    for param_name in params_map:
        param_values = params_map[param_name]
        if len(param_values) == 1:
            best_param[param_name] = param_values[0]
            continue

        tmp = Model()
        print 'select best {} for model {}'.format(param_name, tmp.__class__.__name__)

        kwparam = {}
        if not orthogonal:
            kwparam = copy.copy(best_param)
        
        best_score = 0
        for value in param_values:
            kwparam[param_name] = value
            mdl_temp = Model(**kwparam)
            score = cross_val_score(mdl_temp, trainX, trainY, cv=folds).mean()
            if score > best_score:
                best_score = score
                best_param[param_name] = value
    
    print 'best param {}'.format(best_param)
    print '*' * 20
    return best_param
#%%
#XGB参数调优
params_map = {
    'max_depth': [2],
    'learning_rate': [0.35],
    'min_child_weight': [6],
    'max_delta_step': [2],
    'base_score': [0.88]
}

xbg_param = engineering_params(XGBClassifier, params_map)

#KNeighborsClassifier参数调优
params_map = {
    'n_neighbors': [7],
    'leaf_size': [5],
    'p': [4]
}

knn_param = engineering_params(KNeighborsClassifier, params_map)

#svc参数调优
params_map = {
    'C': [0.075],
    'cache_size': [200]
}

svc_param = engineering_params(SVC, params_map)

ExtC = ExtraTreesClassifier()
params_map = {
    'max_depth': [None],
    'max_features': [1, 3, 10],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [False],
    'n_estimators' :[100,300],
    'criterion': ['gini']
}

gsExtC = GridSearchCV(ExtC, param_grid = params_map, cv=KFold(n_splits=3), scoring='accuracy', n_jobs= 4, verbose = 1)
gsExtC.fit(trainX, trainY)
extc_param = gsExtC.best_params_
print 'extc best param {} at score '.format(extc_param)

DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
params_map = {
    'base_estimator__criterion' : ['gini', 'entropy'],
    'base_estimator__splitter' :   ['best', 'random'],
    'algorithm' : ['SAMME','SAMME.R'],
    'n_estimators' :[1,2],
    'learning_rate':  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]
}

gsadaDTC = GridSearchCV(adaDTC,param_grid = params_map, cv=KFold(n_splits=3), scoring='accuracy', n_jobs= 4, verbose = 1)
gsadaDTC.fit(trainX, trainY)
ada_param = gsadaDTC.best_estimator_
print 'ada best param {} at score '.format(ada_param)
#%%
mdl_rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
mdl_xgb = XGBClassifier(**xbg_param)
mdl_knn = KNeighborsClassifier(**knn_param)
mdl_svc = SVC(**svc_param)
mdl_extc = gsExtC.best_estimator_
mdl_ada = gsadaDTC.best_estimator_

mdl_voting = VotingClassifier([('rfc', mdl_rfc), ('xgb', mdl_xgb), ('extc', mdl_extc), ('ada', mdl_ada)])
mdl_stacking = StackingClassifer([mdl_extc, mdl_xgb, mdl_rfc], mdl_ada)
mdl = mdl_voting

mdl.fit(trainX, trainY)
print 'test shape = {}'.format(testX.shape)
mdl_pred = mdl.predict(testX)

print 'score = {}'.format(mdl.score(trainX, trainY))

result = pd.DataFrame({
    'PassengerId': testId,
    'Survived': mdl_pred
})

result.to_csv('./titanic/submission/result.csv', index=False)
print 'done'