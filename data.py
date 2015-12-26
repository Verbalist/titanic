__author__ = 'verbalist'
from pandas import read_csv

import seaborn
from sklearn.preprocessing import normalize

dataset = read_csv('train.csv')

target = dataset.Survived

