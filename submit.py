# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import numpy as np
import pandas as pd
from datetime import datetime as dt
from utils import create_filelist

#
#
#

n = 30

sequense_file = "../submissions/sequence-{}.csv".format(n)
koefs = list(np.loadtxt(sequense_file))

df = pd.read_csv('test-set-{}-smaller.csv'.format(n))
df.drop(['match'], axis=1, inplace=True)

test_path = '../data/test_smaller/'
test_files = create_filelist(test_path)

final = []

for fl in test_files:

    subs = df[df['setId'] == int(fl)]
    ids = subs['ID'].tolist()

    maxval = 0.0
    maxidx = 0

    for idx in ids:
        if koefs[idx] > maxval:
            maxval = koefs[idx]
            maxidx = idx

    seq = subs.loc[subs['ID'] == maxidx, 'files'].values[0]

    final.append(dict(
        setId=fl,
        day=seq,
    ))


pd.DataFrame(final).to_csv(
    "../submissions/submit-FINAL-%s.csv" % dt.now().strftime("%Y-%m-%d-%H-%M"),
    sep=",",
    index=False
)
