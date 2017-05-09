#%%
import pandas as pd

from sklearn.linear_model import LogisticRegression

train_dt = pd.read_csv('titanic/train.csv')
test_dt = pd.read_csv('titanic/test.csv')

#0 female 1 male 2 child
def define_person(info):
    age, sex = info
    return 2 if age < 16 else (0 if sex == 'female' else 1)

def process_data(data):
#fill na
    data['Age'] = data['Age'].fillna(int(data['Age'].median()))
    data['Fare'] = data['Fare'].fillna(int(data['Fare'].median()))
#create new member
    data['Person'] = data[['Age', 'Sex']].apply(define_person, axis=1)
#drop columns
    data = data.drop(['Sex'], axis=1)
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

#logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
def validation_curve_model(X, Y, model, param_name, parameters, cv, ylim, log=True):
    train_scores, test_scores = validation_curve(model, X, Y, param_name=param_name, param_range=parameters, cv=cv, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title('Validation curve')
    plt.fill_between(parameters, parameters, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(parameters, parameters, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    
    if log == True:
        plt.semilogx(parameters, train_scores_mean, 'o-', color='r', label='Train score')
        plt.semilogx(parameters, test_scores_mean, 'o-', color='g', label='Cross validation score')
    else:
        plt.plot(parameters, train_scores_mean, 'o-', color='r', label='Train score')
        plt.plot(parameters, test_scores_mean, 'o-', color='g', label='Cross validation score')

    if ylim is not None:
        plt.ylim(*ylim)

    plt.ylabel('Score')
    plt.xlabel('Parameter C')
    plt.legend(loc='best')

    return plt

def Learning_curve_model(X, Y, model, cv, train_sizes):
    plt.figure()
    plt.title('Learning curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(parameters, parameters, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(parameters, parameters, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc='best')
    return plt

