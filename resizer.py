# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import cv2
from utils import create_filelist


def resizzze(set_type):
    """
    Resize image to smaller size
    :param set_type: set type ('train' or 'test')
    :return:
    """
    path = '../data/{}_sm/'.format(set_type)
    path_new = '../data/{}_smaller/'.format(set_type)

    files = create_filelist(path)

    for fl in files:

        for idx in range(1, 6):

            filename = 'set{}_{}.jpeg'.format(fl, idx)

            filename_from = path + filename

            image = cv2.imread(filename_from)

            # new dimensions
            r = 750.0 / image.shape[1]
            dim = (750, int(image.shape[0] * r))
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            print path + filename

            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path_new + filename, resized)


if __name__ == '__main__':

    print 'Resize train set...'
    resizzze('train')

    print 'Resize test set...'
    resizzze('test')
