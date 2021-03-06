#%%
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as Matrix

train_dt = pd.read_csv('titanic/train.csv')
test_dt = pd.read_csv('titanic/test.csv')

test_pid = test_dt['PassengerId']

corrmat = train_dt.corr()
sns.heatmap(corrmat, vmax=8, square=True)

#0 female 1 male 2 child
def define_person(info):
    age, sex = info
    return 2 if age < 16 else (0 if sex == 'female' else 1)

def process_data(data):
#fill na
    data['Age'] = data['Age'].fillna(int(data['Age'].median()))
    data['Fare'] = data['Fare'].fillna(int(data['Fare'].median()))
    data['Sex'] = data['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
#create new member
    #data['Person'] = data[['Age', 'Sex']].apply(define_person, axis=1)
#drop columns
    #data = data.drop(['Sex'], axis=1)
    data = data.drop(['Cabin'], axis=1)
    data = data.drop(['Embarked'], axis=1)
    data = data.drop(['PassengerId'], axis=1)
    data = data.drop(['Name'], axis=1)
    data = data.drop(['Ticket'], axis=1)
    return data

train_dt = process_data(train_dt)
test_dt = process_data(test_dt)

X_train = train_dt.drop('Survived', axis=1)
Y_train = train_dt['Survived']
X_test = test_dt.copy()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = ({
    # 'logreg': LogisticRegression(),
    # 'svc': SVC(),
    'random forest': RandomForestClassifier(),
    # 'knn': KNeighborsClassifier(n_neighbors=3),
    # 'GaussinNB': GaussianNB()
})

paramsDef = ({
    # 'logreg': { 'name': 'C', 'data': [1, 3, 10, 30, 100, 300] },
    # 'svc': { 'name': 'C', 'data': [1, 3, 10, 30, 100, 300] },
    'random forest': { 'name': 'n_estimators', 'data': [10, 20, 50, 100, 200] },
    # 'knn': { 'name': 'leaf_size', 'data': [30, 60, 90, 120, 150, 180] },
    # 'GaussinNB': { 'name': 'priors', 'data': [None] }
})

bestScore = 0

for name in models:
    model = models[name]
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    score = model.score(X_train, Y_train)
    print('%s = %f'%(name, score))

    if score > bestScore:
        bestScore = score
        bestPred = Y_pred

#%%
#use cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

bestScore = 0
bestModel = None
for name in models:
    model = models[name]
    scores = cross_val_score(model, X_train, Y_train, cv=3, scoring='f1')
    mean = scores.mean()
    std = scores.std()
    print('mean = %0.2f, std = %0.2f'%(mean, std))

    if mean > bestScore:
        bestScore = mean
        bestModel = name

print('best model = %s, score = %0.2f'%(bestModel, bestScore))

#%%
#plot curve
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

def plot_validation_curve(title, params, paramName, t_scores, t_stds, v_scores, v_stds):
    fig = plt.figure()
    plt.title(title)
    x_axis = range(len(params))
    plt.fill_between(x_axis, t_scores - t_stds, t_scores + t_stds, alpha=0.1, color='r')
    plt.fill_between(x_axis, v_scores - v_stds, v_scores + v_stds, alpha=0.1, color='g')
    plt.plot(x_axis, t_scores, 'o-', color='r', label='Training Score')
    plt.plot(x_axis, v_scores, 'o-', color='g', label='Cross Validation Score')
    plt.ylabel('Score')
    plt.xlabel(paramName)
    plt.legend(loc='best')
    ax = fig.add_subplot(111)
    for xy in zip(x_axis, t_scores):
        ax.annotate(str(params[xy[0]]), xy=xy)
    
    for xy in zip(x_axis, v_scores):
        ax.annotate(str(params[xy[0]]), xy=xy)

for modelName in paramsDef:
    print modelName
    paramDef = paramsDef[modelName]
    paramName = paramDef['name']
    paramData = paramDef['data']
    tScores, vScores = validation_curve(models[modelName], X_train, Y_train, paramName, paramData)
    tScoresMean = np.mean(tScores, axis=1)
    tScoresStd = np.std(tScores, axis=1)
    vScoresMean = np.mean(vScores, axis=1)
    vScoresStd = np.std(vScores, axis=1)
    plot_validation_curve('%s Validation Curve'%modelName, paramData, paramName, tScoresMean, tScoresStd, vScoresMean, vScoresStd)

#%%
#plot learning curve
from sklearn.model_selection import learning_curve

models = ({
    # 'logreg': LogisticRegression(),
    # 'svc': SVC(),
    'random forest': RandomForestClassifier(n_estimators=100, criterion='entropy'),
    # 'knn': KNeighborsClassifier(n_neighbors=3),
    # 'GaussinNB': GaussianNB()
})

def plot_learning_curve(title, ticks, t_scores, t_stds, v_scores, v_stds):
    fig = plt.figure()
    plt.title(title)
    plt.fill_between(ticks, t_scores - t_stds, t_scores + t_stds, alpha=0.1, color='r')
    plt.fill_between(ticks, v_scores - v_stds, v_scores + v_stds, alpha=0.1, color='g')
    plt.plot(ticks, t_scores, 'o-', color='r', label='Training Score')
    plt.plot(ticks, v_scores, 'o-', color='g', label='Test Score')
    plt.ylabel('Score')
    plt.xlabel('Ticks')
    plt.legend(loc='best')
    ax = fig.add_subplot(111)
    for xy in zip(ticks, t_scores):
        ax.annotate(str(round(xy[1], 3)), xy=xy)
    
    for xy in zip(ticks, v_scores):
        ax.annotate(str(round(xy[1], 3)), xy=xy)

for name in models:
    model = models[name]
    tSizes, tScores, vScores = learning_curve(model, X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 5))
    tScoresMean = np.mean(tScores, axis=1)
    tScoresStd = np.std(tScores, axis=1)
    vScoresMean = np.mean(vScores, axis=1)
    vScoresStd = np.std(vScores, axis=1)
    plot_learning_curve('%s Learning Curve'%name, tSizes, tScoresMean, tScoresStd, vScoresMean, vScoresStd)
#%%
model = RandomForestClassifier(n_estimators=100, criterion='entropy')
model.fit(X_train, Y_train)
pred = model.predict(X_test)
submission = Matrix({
    'PassengerId': test_pid,
    'Survived': pred
})

submission.to_csv('titanic/titanic.csv', index=False)