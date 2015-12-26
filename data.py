__author__ = 'verbalist'
from pandas import read_csv, DataFrame

import seaborn
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
from  csv import DictWriter
dataset = read_csv('train.csv', index_col='PassengerId')
# fieldnames = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
#               'Fare', 'Cabin', 'Embarked']
#
# train_wr = DictWriter(open('X.csv', 'w'), fieldnames)
# test_wr = DictWriter(open('X_t.csv', 'w'), fieldnames)


# for i, x in enumerate(dataset):
#      if i > 2/3*len(dataset):
#           test_wr.writerow()

# test = read_csv('test.csv')
train = dataset[:int(len(dataset)/3*2)]
test = dataset[int(len(dataset)/3*2):]
target = dataset.Survived[:int(len(dataset)/3*2)]
target_test = dataset.Survived[int(len(dataset)/3*2):]

avg_age = sum([x for x in dataset.Age if not np.isnan(x)])/len([x for x in dataset.Age if not np.isnan(x)])
avg_fare = sum([x for x in dataset.Fare if not np.isnan(x)])/len([x for x in dataset.Fare if not np.isnan(x)])


X = [
     [x if not np.isnan(x) else -1 for x in train.Pclass],
     [0 if x =='male' else 1 if x == 'female' else -1 for x in train.Sex],
     [x if not np.isnan(x) else avg_age for x in train.Age],
     [x if not np.isnan(x) else -1 for x in train.SibSp],
     [x if not np.isnan(x) else -1 for x in train.Parch],
     [avg_fare if np.isnan(x) else 263 if x > 300 else x for x in train.Fare],
     [0 if x =='C' else 1 if x == 'Q' else 2 for x in train.Embarked]]

X_t = [
       [x if not np.isnan(x) else -1 for x in test.Pclass],
       [0 if x =='male' else 1 if x == 'female' else -1 for x in test.Sex],
       [x if not np.isnan(x) else avg_age for x in test.Age],
       [x if not np.isnan(x) else -1 for x in test.SibSp],
       [x if not np.isnan(x) else -1 for x in test.Parch],
       [avg_fare if np.isnan(x) else 263 if x > 300 else x for x in test.Fare],
       [0 if x =='C' else 1 if x == 'Q' else 2 for x in test.Embarked]]

X = DataFrame(np.array(X).T)
X_t = DataFrame(np.array(X_t).T)

clf = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001)
clf.fit(X, target)

print(metrics.classification_report(target_test, clf.predict(X_t)))

width = 0.35
plt.bar(np.arange(len(clf.feature_importances_)), clf.feature_importances_, width=0.35)
use_field =  ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
plt.xticks(np.arange(len(clf.feature_importances_)) + width/2., use_field)

plt.show()
