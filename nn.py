# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import normalize
from utils import order_importance, create_filelist

seed = 77
np.random.seed(seed)

train_path = '../data/train-set-30-smaller.csv'
test_path = '../data/test-set-30-smaller.csv'

# load train
train = pd.read_csv(train_path, index_col='ID')
Y = train['right']
X = np.genfromtxt(train['match'])

# order importance
new_X = order_importance(X)

# normalization
X = normalize(new_X, norm='l1')

# Test set
test = pd.read_csv(test_path, index_col='ID')
test_X = np.genfromtxt(test['match'])
test_X_new = order_importance(test_X)

# normalization
test_X = normalize(test_X_new, norm='l1')

# model
model = Sequential()
model.add(Dense(400, input_dim=400, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, init='normal', activation='sigmoid'))
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.33, nb_epoch=300, batch_size=5, verbose=1)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# prediction sequence
prediction = model.predict(test_X)
np.savetxt(
    "../submissions/sequence-30.csv",
    prediction
)
