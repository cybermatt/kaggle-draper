# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def plotMatches(matches, grid_col=2, grid_row=5):
    """
    Plot histogram
    :param matches: dict with matches
    :param gridCol: plot columns
    :param gridRow: plot rows
    :return:
    """
    plt.figure(1)
    i = 1
    for name in matches:
        plt.subplot(grid_col, grid_row, i)
        plt.hist(matches[name], bins=50)
        plt.title(name)
        plt.grid(True)
        i += 1

    plt.show()


def match_bf(img1path, img2path, num_keypoints=1000):
    """
    Match two images using ORB
    :param img1path:
    :param img2path:
    :return:
    """
    orb = cv2.ORB(num_keypoints, 1.2)

    img_from = cv2.imread(img1path)
    img_to = cv2.imread(img2path)

    # comparision
    (kp1, des1) = orb.detectAndCompute(img_from, None)
    (kp2, des2) = orb.detectAndCompute(img_to, None)

    # matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # sort matches
    matches = sorted(matches, key=lambda val: val.distance)

    return matches


def match_knn(img1path, img2path, knn=2):
    """
    Match two images using KNN
    :param img1path:
    :param img2path:
    :return:
    """
    orb = cv2.ORB()

    img_from = cv2.imread(img1path)
    img_to = cv2.imread(img2path)

    # comparision
    (kp1, des1) = orb.detectAndCompute(img_from, None)
    (kp2, des2) = orb.detectAndCompute(img_to, None)

    # matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=knn)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good


def match_flann(img1path, img2path):
    """
    Flann matcher
    :param img1path:
    :param img2path:
    :return:
    """
    orb = cv2.ORB(1000, 1.2)

    img_from = cv2.imread(img1path)
    img_to = cv2.imread(img2path)

    (kp1, des1) = orb.detectAndCompute(img_from, None)
    (kp2, des2) = orb.detectAndCompute(img_to, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)     # or empty dict

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    return matches


def create_filelist(mypath):
    """
    Generate set of files id
    :param mypath:
    :return:
    """
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    fileslist = []
    for f in onlyfiles:
        fidx = f.split('_')[0].replace('set', '')
        fileslist.append(fidx)

    return sorted(set(fileslist))


def order_importance(dataset):

    new_X = []
    for row in dataset:
        nrow = []
        for idx, elem in enumerate(row):
            koeff = (100 / ((idx % 100) + 0.01) ** 2)
            nrow.append(elem * koeff)

        new_X.append(nrow)

    return new_X
