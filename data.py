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

# ticket_code = [int(x.split(' ')[-1]) if str(x) != 'LINE' else -1 for x in train.Ticket]
# print(ticket_code)
ticket_name_1 = [y.split('/')[0] if len(y.split('/')) > 0 else '0' for y in
                            (''.join(x.split(' ')[:-1]) if str(x) != 'nan' else '' for x in dataset.Ticket)]
ticket_name_2 = [y.split('/')[1] if len(y.split('/')) > 1 else '0' for y in
                             (''.join(x.split(' ')[:-1]) if str(x) != 'nan' else '' for x in dataset.Ticket)]

ticket_name_1_class = {x: i for i, x in enumerate(set(ticket_name_1))}
ticket_name_2_class = {x: i for i, x in enumerate(set(ticket_name_2))}
# print([ticket_name_1_class[x] for x in ticket_name_1])

# print(ticket_name_1_class)
# print(ticket_name_2_class)
# print('######################################')


avg_age = sum([x for x in dataset.Age if not np.isnan(x)])/len([x for x in dataset.Age if not np.isnan(x)])
avg_fare = sum([x for x in dataset.Fare if not np.isnan(x)])/len([x for x in dataset.Fare if not np.isnan(x)])


X = [
     [x if not np.isnan(x) else -1 for x in train.Pclass],
     [0 if x =='male' else 1 if x == 'female' else -1 for x in train.Sex],
     normalize([x if not np.isnan(x) else avg_age for x in train.Age])[0],
     [x if not np.isnan(x) else -1 for x in train.SibSp],
     [x if not np.isnan(x) else -1 for x in train.Parch],
     [ticket_name_1_class[x] for x in ticket_name_1[:int(len(dataset)/3*2)]],
     [ticket_name_2_class[x] for x in ticket_name_2[:int(len(dataset)/3*2)]],
     # [1 if str(x) != 'LINE' else -1 for x in train.Ticket],
     normalize([avg_fare if np.isnan(x) else 263 if x > 300 else x for x in train.Fare])[0],
     [0 if x =='C' else 1 if x == 'Q' else 2 for x in train.Embarked]]

X_t = [
       [x if not np.isnan(x) else -1 for x in test.Pclass],
       [0 if x =='male' else 1 if x == 'female' else -1 for x in test.Sex],
       normalize([x if not np.isnan(x) else avg_age for x in test.Age])[0],
       [x if not np.isnan(x) else -1 for x in test.SibSp],
       [x if not np.isnan(x) else -1 for x in test.Parch],
       [ticket_name_1_class[x] for x in ticket_name_1[int(len(dataset)/3*2):]],
       [ticket_name_2_class[x] for x in ticket_name_2[int(len(dataset)/3*2):]],
       # [1 if str(x) != 'LINE' else -1 for x in test.Ticket],
       normalize([avg_fare if np.isnan(x) else 263 if x > 300 else x for x in test.Fare])[0],
       [0 if x =='C' else 1 if x == 'Q' else 2 for x in test.Embarked]]

X = DataFrame(np.array(X).T)
X_t = DataFrame(np.array(X_t).T)

