'''hello world'''

#%%
import copy
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, Lasso
from xgboost.sklearn import XGBClassifier

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
#%%
def engineering_params(Model, params_map, orthogonal=False):
    folds = KFold(n_splits=3)
    best_param = {}
    for param_name in params_map:
        param_values = params_map[param_name]
        tmp = Model()
        print 'select best {} for model {}'.format(param_name, tmp.__class__.__name__)

        kwparam = {}
        if not orthogonal:
            kwparam = copy.copy(best_param)
        
        best_score = 0;
        for value in param_values:
            kwparam[param_name] = value
            mdl_temp = Model(**kwparam)
            score = cross_val_score(mdl_temp, trainX, trainY, cv=folds).mean()
            if score > best_score:
                best_score = score
                best_param[param_name] = value
    
    print 'best param {}'.format(best_param)
    return best_param
#XGB参数调优
params_map = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.03, 0.1, 0.3, 1],
    'gamma': [0, 0.01, 0.03, 0.1, 0.3, 1],
    'min_child_weight': [1, 3, 5, 7],
    'max_delta_step': [0, 1, 2, 3],
    'reg_alpha': [0, 0.01, 0.03, 0.1, 0.3, 1],
    'reg_lambda': [0, 0.01, 0.03, 0.1, 0.3, 1],
    'scale_pos_weight': [0, 0.01, 0.03, 0.1, 0.3, 1],
    'base_score': [0.1, 0.3, 0.5, 0.7, 0.9]
}

xbg_param = engineering_params(XGBClassifier, params_map)
#%%
mdl_rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
mdl_xgb = XGBClassifier(**xbg_param)
mdl_enet = ElasticNet()
mdl_lasso = Lasso()

mdl_stacking = StackingClassifer([mdl_xgb, mdl_enet, mdl_lasso], mdl_rfc)

mdl_stacking.fit(trainX, trainY)
mdl_stacking_pred = mdl_stacking.predict(testX)

# print mdl_rfc.score(trainX, trainY)

result = pd.DataFrame({
    'PassengerId': testId,
    'Survived': mdl_stacking_pred
})

result.to_csv('./titanic/submission/result.csv', index=False)
print 'done'