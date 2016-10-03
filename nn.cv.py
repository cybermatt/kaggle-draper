# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize

from utils import order_importance

#

seed = 7
np.random.seed(seed)

# load data
file_name = '../data/train-set-30-smaller.csv'
train = pd.read_csv(file_name, index_col='ID')
Y = train['right']
X = np.genfromtxt(train['match'])

# order importance
new_X = order_importance(X)

# normalization
X = normalize(new_X, norm='l1')

#
#   Splitting and fit
#

kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)

cv_scores = []

for i, (train, test) in enumerate(kfold):

    model = Sequential()
    model.add(Dense(400, input_dim=400, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X[train], Y[train], nb_epoch=200, batch_size=10, verbose=0)

    scores = model.evaluate(X[test], Y[test], verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv_scores.append(scores[1] * 100)

print "%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores))
