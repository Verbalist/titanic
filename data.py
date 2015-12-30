__author__ = 'verbalist'
from pandas import read_csv, DataFrame
import numpy as np
from sklearn.preprocessing import normalize
import itertools as it
dataset = read_csv('train.csv', index_col='PassengerId')

train = dataset[:int(len(dataset)/3*2)]
test = dataset[int(len(dataset)/3*2):]
target = dataset.Survived[:int(len(dataset)/3*2)]
target_test = dataset.Survived[int(len(dataset)/3*2):]

ticket_name_1 = [y.split('/')[0] if len(y.split('/')) > 0 else '0' for y in
                            (''.join(x.split(' ')[:-1]) if str(x) != 'nan' else '' for x in dataset.Ticket)]
ticket_name_2 = [y.split('/')[1] if len(y.split('/')) > 1 else '0' for y in
                             (''.join(x.split(' ')[:-1]) if str(x) != 'nan' else '' for x in dataset.Ticket)]

ticket_name_1_class = {x: i for i, x in enumerate(set(ticket_name_1))}
ticket_name_2_class = {x: i for i, x in enumerate(set(ticket_name_2))}
test_dataset = read_csv('test.csv', index_col='PassengerId')
name_class = {x: i for i, x in enumerate(set([x.split(',')[1].split('.')[0].strip() for x in np.hstack((dataset.Name, test_dataset.Name))]))}
avg_age_class = DataFrame(DataFrame(np.array([[x, name_class[y.split(',')[1].split('.')[0].strip()]] for x, y in
        zip(dataset.Age, dataset.Name)]), columns=['age', 'title']).groupby('title')['age'].mean())
avg_fare_class = dataset.groupby('Pclass')['Fare'].mean()

train_title_mas = [name_class[x.split(',')[1].split('.')[0].strip()] for x in train.Name]
test_title_mas = [name_class[x.split(',')[1].split('.')[0].strip()] for x in test.Name]
dataset_title_mas = [name_class[x.split(',')[1].split('.')[0].strip()] for x in dataset.Name]
test_dataset_title_mas = [name_class[x.split(',')[1].split('.')[0].strip()] for x in test_dataset.Name]

title_class = {x.split(',')[1].split('.')[0].strip(): i if x.split(',')[1].split('.')[0].strip()
    in ("Mrs", "Mr", "Miss", "Master") else -1 for i, x in enumerate(np.hstack((dataset.Name, test_dataset.Name)))}


# print(avg_age_class['avg_age_class']==0])
# print(avg_age_class['title'])
# print(avg_age_class[avg_age_class['title'] == 1])

# family = [x.split(',')[0] for x in dataset.Name]
# title = [1 if x.split(',')[1].split('.')[0].strip() in ('Dona', 'Lady', 'the Countess', 'Jonkheer', 'Capt', 'Don', 'Major', 'Sir') else 0 for x in dataset.Name]
# print(len(dataset.Name), len(family))
# print(family)
#

X = [
     [x if not np.isnan(x) else -1 for x in train.Pclass],
     [name_class[x.split(',')[1].split('.')[0].strip()] for x in train.Name],
     [0 if x =='male' else 1 if x == 'female' else -1 for x in train.Sex],
     normalize([x if not np.isnan(x) else avg_age_class.age[train_title_mas[i]] for i, x in enumerate(train.Age)])[0],
     [title_class[x.split(',')[1].split('.')[0].strip()] for x in train.Name],
     [x for x in train.SibSp + train.Parch],
     [1 if x != 0 else 0 for x in train.SibSp + train.Parch],
     [1 if x == 3 and y.split(',')[1].split('.')[0].strip() == 'Mr' else 0 for x,y in zip(train.Pclass, train.Name)],
     [ticket_name_1_class[x] for x in ticket_name_1[:int(len(dataset)/3*2)]],
     [ticket_name_2_class[x] for x in ticket_name_2[:int(len(dataset)/3*2)]],
     [1 if not str(x) == 'nan' else 0 for x in train.Cabin],
     normalize([avg_fare_class[y] if x == 0.0 else 263 if x > 300 else x for x, y in zip(train.Fare, train.Pclass)])[0],
     [0 if x == 'C' else 1 if x == 'Q' else 2 for x in train.Embarked]]
#
X_t = [
       [x if not np.isnan(x) else -1 for x in test.Pclass],
       [name_class[x.split(',')[1].split('.')[0].strip()] for x in test.Name],
       [0 if x =='male' else 1 if x == 'female' else -1 for x in test.Sex],
       normalize([x if not np.isnan(x) else avg_age_class.age[test_title_mas[i]] for i, x in enumerate(test.Age)])[0],
       [title_class[x.split(',')[1].split('.')[0].strip()] for x in test.Name],
       [x for x in test.SibSp + test.Parch],
       [1 if x != 0 else 0 for x in test.SibSp + test.Parch],
       [1 if x == 3 and y.split(',')[1].split('.')[0].strip() == 'Mr' else 0 for x,y in zip(test.Pclass, test.Name)],
       [ticket_name_1_class[x] for x in ticket_name_1[int(len(dataset)/3*2):]],
       [ticket_name_2_class[x] for x in ticket_name_2[int(len(dataset)/3*2):]],
       [1 if not str(x) == 'nan' else 0 for x in test.Cabin],
       normalize([avg_fare_class[y] if x == 0.0 else 263 if x > 300 else x for x, y in zip(test.Fare, test.Pclass)])[0],
       [0 if x == 'C' else 1 if x == 'Q' else 2 for x in test.Embarked]]

X_full = [
       [x if not np.isnan(x) else -1 for x in dataset.Pclass],
       [name_class[x.split(',')[1].split('.')[0].strip()] for x in dataset.Name],
       [0 if x =='male' else 1 if x == 'female' else -1 for x in dataset.Sex],
       normalize([x if not np.isnan(x) else avg_age_class.age[dataset_title_mas[i]] for i, x in enumerate(dataset.Age)])[0],
       [title_class[x.split(',')[1].split('.')[0].strip()] for x in dataset.Name],
       [x for x in dataset.SibSp + dataset.Parch],
       [1 if x != 0 else 0 for x in dataset.SibSp + dataset.Parch],
       [1 if x == 3 and y.split(',')[1].split('.')[0].strip() == 'Mr' else 0 for x,y in zip(dataset.Pclass, dataset.Name)],
       # [ticket_name_1_class[x] for x in ticket_name_1],
       # [ticket_name_2_class[x] for x in ticket_name_2],
       [1 if not str(x) == 'nan' else 0 for x in dataset.Cabin],
       normalize([avg_fare_class[y] if x == 0.0 else 263 if x > 300 else x for x, y in zip(dataset.Fare, dataset.Pclass)])[0],
       [0 if x == 'C' else 1 if x == 'Q' else 2 for x in dataset.Embarked]]

test_predict = [
    [x if not np.isnan(x) else -1 for x in test_dataset.Pclass],
       [name_class[x.split(',')[1].split('.')[0].strip()] for x in test_dataset.Name],
       [0 if x =='male' else 1 if x == 'female' else -1 for x in test_dataset.Sex],
       normalize([x if not np.isnan(x) else avg_age_class.age[test_dataset_title_mas[i]] for i, x in enumerate(test_dataset.Age)])[0],
       [title_class[x.split(',')[1].split('.')[0].strip()] for x in test_dataset.Name],
       [x for x in test_dataset.SibSp + test_dataset.Parch],
       [1 if x != 0 else 0 for x in test_dataset.SibSp + test_dataset.Parch],
       [1 if x == 3 and y.split(',')[1].split('.')[0].strip() == 'Mr' else 0 for x,y in zip(test_dataset.Pclass, test_dataset.Name)],
       # [ticket_name_1_class[x] for x in ticket_name_1],
       # [ticket_name_2_class[x] for x in ticket_name_2],
       [1 if not str(x) == 'nan' else 0 for x in test_dataset.Cabin],
       normalize([avg_fare_class[y] if x == 0.0 or np.isnan(x) else 263 if x > 300 else x for x, y in zip(test_dataset.Fare, test_dataset.Pclass)])[0],
       [0 if x == 'C' else 1 if x == 'Q' else 2 for x in test_dataset.Embarked]]

X = DataFrame(np.array(X).T)
X_t = DataFrame(np.array(X_t).T)
X_full = DataFrame(np.array(X_full).T)
X_f_t = DataFrame(np.array(test_predict).T)


# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import metrics
# clf = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001)
# clf.fit(X, target)
# print(metrics.classification_report(target_test, clf.predict(X_t)))
#

# # clf = GradientBoostingClassifier(n_estimators=350, learning_rate=0.001)
# clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.001)
# clf.fit(X_full, dataset.Survived)
# # print(metrics.classification_report(target_test, clf.predict(X_t)))
# from csv import DictWriter
# w = DictWriter(open('solve.csv', 'w'), fieldnames=['PassengerId', 'Survived'])
# w.writeheader()
# for i, x in enumerate(clf.predict(X_f_t)):
#     w.writerow({'PassengerId': i + 892, 'Survived': x})
