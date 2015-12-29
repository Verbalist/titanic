__author__ = 'verbalist'

from sklearn import metrics
import seaborn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
from data import X_t, target, target_test, X \
    ,X_full, dataset, X_f_t


clf = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001)
# clf.fit(X, target)
# print(metrics.classification_report(target_test, clf.predict(X_t)))


# clf = RandomForestClassifier(n_jobs=4, n_estimators=10000, min_samples_leaf=3)
# clf.fit(X, target)
# print(metrics.classification_report(target_test, clf.predict(X_t)))

clf.fit(X_full, dataset.Survived)
from csv import DictWriter
w = DictWriter(open('solve.csv', 'w'), fieldnames=['PassengerId', 'Survived'])
w.writeheader()
for i, x in enumerate(clf.predict(X_f_t)):
    w.writerow({'PassengerId': i + 892, 'Survived': x})




width = 0.35
plt.bar(np.arange(len(clf.feature_importances_)), clf.feature_importances_, width=0.35)
use_field = ['Pclass',
             'Name',
             'Sex',
             'Age',
             'Title',
             'SibSp+Parch',
             'is_SibSp',
             'is_3cl_man',
             'T1',
             'T2',
             'Cabin',
             'Fare',
             'Embarked']
plt.xticks(np.arange(len(clf.feature_importances_)) + width/2., use_field)

plt.show()
