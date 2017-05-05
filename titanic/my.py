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