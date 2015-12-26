__author__ = 'verbalist'
from pandas import read_csv, DataFrame
import numpy as np
from sklearn.preprocessing import normalize

dataset = read_csv('train.csv', index_col='PassengerId')
train = dataset[:int(len(dataset)/3*2)]
test = dataset[int(len(dataset)/3*2):]
target = dataset.Survived[:int(len(dataset)/3*2)]
target_test = dataset.Survived[int(len(dataset)/3*2):]

avg_age = sum([x for x in dataset.Age if not np.isnan(x)])/len([x for x in dataset.Age if not np.isnan(x)])
avg_fare = sum([x for x in dataset.Fare if not np.isnan(x)])/len([x for x in dataset.Fare if not np.isnan(x)])


X = [
     [x if not np.isnan(x) else -1 for x in train.Pclass],
     [0 if x =='male' else 1 if x == 'female' else -1 for x in train.Sex],
     normalize([x if not np.isnan(x) else avg_age for x in train.Age])[0],
     [x if not np.isnan(x) else -1 for x in train.SibSp],
     [x if not np.isnan(x) else -1 for x in train.Parch],
     normalize([avg_fare if np.isnan(x) else 263 if x > 300 else x for x in train.Fare])[0],
     [0 if x =='C' else 1 if x == 'Q' else 2 for x in train.Embarked]]

X_t = [
       [x if not np.isnan(x) else -1 for x in test.Pclass],
       [0 if x =='male' else 1 if x == 'female' else -1 for x in test.Sex],
       normalize([x if not np.isnan(x) else avg_age for x in test.Age])[0],
       [x if not np.isnan(x) else -1 for x in test.SibSp],
       [x if not np.isnan(x) else -1 for x in test.Parch],
       normalize([avg_fare if np.isnan(x) else 263 if x > 300 else x for x in test.Fare])[0],
       [0 if x =='C' else 1 if x == 'Q' else 2 for x in test.Embarked]]

X = DataFrame(np.array(X).T)
X_t = DataFrame(np.array(X_t).T)

