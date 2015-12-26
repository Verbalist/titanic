__author__ = 'verbalist'

from sklearn import metrics
import seaborn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
from data import X_t, target, target_test, X

clf = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001)
clf.fit(X, target)

print(metrics.classification_report(target_test, clf.predict(X_t)))

width = 0.35
plt.bar(np.arange(len(clf.feature_importances_)), clf.feature_importances_, width=0.35)
use_field = ['Pclass',
             'Sex',
             'Age',
             'SibSp',
             'Parch',
             'T1',
             'T2',
             # 'T number',
             'Fare',
             'Embarked']
plt.xticks(np.arange(len(clf.feature_importances_)) + width/2., use_field)

plt.show()
