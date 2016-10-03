# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import itertools
import naive_random
import pandas as pd
from utils import create_filelist, match_bf


#
#
#

def generate_sequence(n=50, orig=[1, 2, 3, 4, 5]):
    """
    Generate n sequence for synthetic set
    :param n:
    :param orig:
    :return:
    """
    ops = []
    for j in itertools.permutations(orig):
        ops.append(j)

    naive_random.shuffle(ops)

    res = ops[:n]
    if (1, 2, 3, 4, 5) not in res:
        res[naive_random.randint(0, n - 1)] = (1, 2, 3, 4, 5)

    return res


def file_matches_seq(path, setId, seq, threshold=100):

    resdes = []

    print seq

    for idx, name in enumerate(seq):

        if idx == 4:        # for last element
            break

        filename_from = path + 'set' + str(setId) + '_' + str(seq[idx]) + '.jpeg'
        filename_to = path + 'set' + str(setId) + '_' + str(seq[idx+1]) + '.jpeg'

        matches = match_bf(filename_from, filename_to, 1000)
        mlist = [match.distance for match in matches if match.distance < threshold]

        deses = {x: 0 for x in range(0, 100)}
        for m in mlist:
            deses[m] += 1

        resdes.extend(deses.values())

    return resdes


def generate_set(set_type, Xnegative=50):

    train_path = '../data/{}_smaller/'.format(set_type)
    train_files = create_filelist(train_path)

    train_set = []

    for fl in train_files:

        seqs = generate_sequence(n=Xnegative)
        for seq in seqs:

            if set_type == 'train':
                if seq == (1, 2, 3, 4, 5):
                    right = 1
                else:
                    right = 0

            matches = file_matches_seq(train_path, fl, seq)

            # sequence to file order
            flist = []
            for n in [1, 2, 3, 4, 5]:
                flist.append(seq.index(n) + 1)

            row = dict(
                setId=fl,
                seq=' '.join([str(i) for i in seq]),
                files=' '.join([str(i) for i in flist]),
                match=' '.join(str(v) for v in matches),
            )

            if set_type == 'train':
                row['right'] = right

            train_set.append(row)

    # save frame
    df = pd.DataFrame(train_set)
    df.index.name = 'ID'
    df.to_csv('../data/{}-set-{}-smaller.csv'.format(set_type, Xnegative))

#
#
#

if __name__ == '__main__':

    print 'Create train set...'
    generate_set('train', 30)

    print 'Create test set...'
    generate_set('test', 30)
