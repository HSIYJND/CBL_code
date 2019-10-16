#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn.tree
import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from linear_mapping_utilities import pixmat_between_two_dates, randmat_between_two_dates

#%% BELOW ARE EXP WITHOUT CDA
def pixmat_KNN_without_CDA_using_unbalanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP1 : pixmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    without CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = pixmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    
    # the split was made in matlab, index starts from 1 instead of 0
    train_indices = trainset["train_indices_splitter"][split] - 1 
    test_indices = trainset["test_indices_splitter"][split] - 1 
    
    # remove all bad bands
    spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    spectra = spectra[:, 2:]
    # remove first two zero-bands
    train_spectra = spectra[train_indices, :]
    train_targets = trainset["labels"][train_indices]
    test_spectra = spectra[test_indices, :]
    same_date_test_target = trainset["labels"][test_indices]
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    same_date_predicts = KNN_classifier.predict(test_spectra)
    same_date_accuracy = round(np.where(same_date_predicts == same_date_test_target)[0].shape[0] / len(same_date_predicts) * 100)
    print("same date KNN:", same_date_accuracy)
    
    # remove all bad bands
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    # remove first two zero-bands
    test_spectra = test_spectra[:, 2:]
    
    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # make prediction, use the whole set
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "pixmat_KNN_without_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy
    
    
def randmat_KNN_without_CDA_using_unbalanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP2 : randmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    without CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = randmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    
    # the split was made in matlab, index starts from 1 instead of 0
    train_indices = trainset["train_indices_splitter"][split] - 1 
    
    # remove all bad bands
    train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    # remove first two zero-bands
    train_spectra = train_spectra[train_indices, 2:]
    train_targets = trainset["labels"][train_indices]
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # remove all bad bands
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    # remove first two zero-bands
    test_spectra = test_spectra[:, 2:]
    
    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "randmat_KNN_without_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def pixmat_KNN_without_CDA_using_balanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP3 : pixmat with KNN using Susan's split
    This function applies PixMat first, and then use KNN to classify.
    without CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = pixmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    
    # the split was made in python, index starts from 0
    train_indices = trainset["train_indices_splitter"][split] 
    
    # bad bands and zero bands are moved when I was making the data file
    train_spectra = trainset["spectra"]
    train_spectra = train_spectra[train_indices, :]
    train_targets = trainset["labels"][train_indices]
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # bad bands and zero bands are moved when I was making the data file
    test_spectra = testset["spectra"]

    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "pixmat_KNN_without_CDA_balance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def randmat_KNN_without_CDA_using_balanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP4 : random with KNN using Susan's split
    This function applies RandMat first, and then use KNN to classify.
    without CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = randmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    
    # the split was made in python, index starts from 0
    train_indices = trainset["train_indices_splitter"][split] 
    
    # bad bands and zero bands are moved when I was making the data file
    train_spectra = trainset["spectra"]
    train_spectra = train_spectra[train_indices, :]
    train_targets = trainset["labels"][train_indices]
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # bad bands and zero bands are moved when I was making the data file
    test_spectra = testset["spectra"]

    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "randmat_KNN_without_CDA_balance_split_pixel_level_accuracy:", accuracy)

    return accuracy

#%% BELOW ARE EXPs WITH CDA

def pixmat_KNN_with_CDA_using_unbalanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP5 : pixmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    with CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = pixmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    
    # the split was made in matlab, index starts from 1 instead of 0
    train_indices = trainset["train_indices_splitter"][split] - 1 
    test_indices = trainset["test_indices_splitter"][split] - 1 
    
    # remove all bad bands
    spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    spectra = spectra[:, 2:]
    # remove first two zero-bands
    train_spectra = spectra[train_indices, :]
    train_targets = trainset["labels"][train_indices]
    
    test_spectra = spectra[test_indices, :]
    same_date_test_target = trainset["labels"][test_indices]
    
    # train knn
    
    
    
    
    # apply CDA
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    # train knn on CDA features
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = clf.transform(test_spectra)
    
    same_date_predicts = KNN_classifier.predict(test_spectra)
    same_date_accuracy = round(np.where(same_date_predicts == same_date_test_target)[0].shape[0] / len(same_date_predicts) * 100)
    print("same date KNN + CDA:", same_date_accuracy)
    
    # remove all bad bands
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    # remove first two zero-bands
    test_spectra = test_spectra[:, 2:]
    
    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction, use the whole set
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "pixmat_KNN_with_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy
    
def randmat_KNN_with_CDA_using_unbalanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP6 : randmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    with CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = randmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    
    # the split was made in matlab, index starts from 1 instead of 0
    train_indices = trainset["train_indices_splitter"][split] - 1 
    
    # remove all bad bands
    train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    # remove first two zero-bands
    train_spectra = train_spectra[train_indices, 2:]
    train_targets = trainset["labels"][train_indices]
    
    # apply CDA on train date
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # remove all bad bands
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    # remove first two zero-bands
    test_spectra = test_spectra[:, 2:]
    
    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "randmat_KNN_with_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def pixmat_KNN_with_CDA_using_balanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP7 : pixmat with KNN using Susan's split
    This function applies PixMat first, and then use KNN to classify.
    with CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = pixmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    
    # the split was made in python, index starts 0
    train_indices = trainset["train_indices_splitter"][split] 
    
    # bad bands and zero bands are moved when I was making the data file
    train_spectra = trainset["spectra"]
    train_spectra = train_spectra[train_indices, :]
    train_targets = trainset["labels"][train_indices]
    
    # apply CDA on train date
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # bad bands and zero bands are moved when I was making the data file
    test_spectra = testset["spectra"]

    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "pixmat_KNN_with_CDA_balance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def randmat_KNN_with_CDA_using_balanced_split(train_date_ = "130411", test_date_ = "140416", split = 0):
    """
    EXP8: random with KNN using Susan's split
    This function applies RandMat first, and then use KNN to classify.
    with CDA
    """
    
    train_date, test_date = train_date_, test_date_
    
    data_path = "../data/"
    
    # A maps test_date back to train_date
    A = randmat_between_two_dates(train_date, test_date, path = data_path)
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    
    # the split was made in python, index starts 0
    train_indices = trainset["train_indices_splitter"][split] 
    
    # bad bands and zero bands are moved when I was making the data file
    train_spectra = trainset["spectra"]
    train_spectra = train_spectra[train_indices, :]
    train_targets = trainset["labels"][train_indices]
    
    # apply CDA on train date
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # bad bands and zero bands are moved when I was making the data file
    test_spectra = testset["spectra"]

    # map test date back to train date
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, 
          "randmat_KNN_with_CDA_balance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy
    
def main(train_date = "130411", test_date = "140416"):
    #%%
    train_date_, test_date_ = train_date, test_date
    print("forward, train on", train_date_, " and test on", test_date_)
    pixmat_unbalanced_no_cda = []
    randmat_unbalanced_no_cda = []
    pixmat_balance_no_cda = []
    randmat_balance_no_cda = []    
    pixmat_unbalance_cda = []
    randmat_unbalance_cda = []
    pixmat_balance_cda = []
    randmat_balance_cda = []
    result = np.zeros((8, 7))
    
    # first two columns: train_date and test_date
    result[:, 0] = int(train_date_) 
    result[:, 1] = int(test_date_) 
    # balanced split, 0: NO, 1 : YES
    result[3:5, 2] = 1
    result[6:8, 2] = 1
    # use CDA? 0 : NO, 1 : YES
    result[4:8, 3] = 1
    # 0: pixmat, 1 : randmat
    result[1:8:2, 4] = 1
    
    for i in range(5):
        print("#### WITHOUT CDA ####################################################################################################")
        pixmat_unbalanced_no_cda.append(pixmat_KNN_without_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_unbalanced_no_cda.append(randmat_KNN_without_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        pixmat_balance_no_cda.append(pixmat_KNN_without_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_balance_no_cda.append(randmat_KNN_without_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        print("#### WITH CDA ###########################################################################################################")
        pixmat_unbalance_cda.append(pixmat_KNN_with_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_unbalance_cda.append(randmat_KNN_with_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        pixmat_balance_cda.append(pixmat_KNN_with_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_balance_cda.append(randmat_KNN_with_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
    
    result[0, 5] = np.mean(pixmat_unbalanced_no_cda)
    result[0, 6] = np.std(pixmat_unbalanced_no_cda)
    result[1, 5] = np.mean(randmat_unbalanced_no_cda)
    result[1, 6] = np.std(randmat_unbalanced_no_cda)
    result[2, 5] = np.mean(pixmat_balance_no_cda)
    result[2, 6] = np.std(pixmat_balance_no_cda)
    result[3, 5] = np.mean(randmat_balance_no_cda)
    result[3, 6] = np.std(randmat_balance_no_cda)
    result[4, 5] = np.mean(pixmat_unbalance_cda)
    result[4, 6] = np.std(pixmat_unbalance_cda)
    result[5, 5] = np.mean(randmat_unbalance_cda)
    result[5, 6] = np.std(randmat_unbalance_cda)
    result[6, 5] = np.mean(pixmat_balance_cda)
    result[6, 6] = np.std(pixmat_balance_cda)
    result[7, 5] = np.mean(randmat_balance_cda)
    result[7, 6] = np.std(randmat_balance_cda)
    
    np.savetxt(train_date_ + "_" + test_date_ + "_linear_mapping_result.csv", result, fmt = "%d", delimiter = "\t", 
               header = "\ttrain_date\ttest_date\tbalanced_split\tapply_CDA\talg\tmean_accu\tstd_accu")
    
    #%%
    train_date_, test_date_ = test_date, train_date
    print("backward, train on", train_date_, " and test on", test_date_)
    pixmat_unbalanced_no_cda = []
    randmat_unbalanced_no_cda = []
    pixmat_balance_no_cda = []
    randmat_balance_no_cda = []    
    pixmat_unbalance_cda = []
    randmat_unbalance_cda = []
    pixmat_balance_cda = []
    randmat_balance_cda = []
    result = np.zeros((8, 7))
    
    # first two columns: train_date and test_date
    result[:, 0] = int(train_date_) 
    result[:, 1] = int(test_date_) 
    # balanced split, 0: NO, 1 : YES
    result[3:5, 2] = 1
    result[6:8, 2] = 1
    # use CDA? 0 : NO, 1 : YES
    result[4:8, 3] = 1
    # 0: pixmat, 1 : randmat
    result[1:8:2, 4] = 1
    
    for i in range(5):
        print("#### WITHOUT CDA ####################################################################################################")
        pixmat_unbalanced_no_cda.append(pixmat_KNN_without_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_unbalanced_no_cda.append(randmat_KNN_without_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        pixmat_balance_no_cda.append(pixmat_KNN_without_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_balance_no_cda.append(randmat_KNN_without_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        print("#### WITH CDA ###########################################################################################################")
        pixmat_unbalance_cda.append(pixmat_KNN_with_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_unbalance_cda.append(randmat_KNN_with_CDA_using_unbalanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        pixmat_balance_cda.append(pixmat_KNN_with_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
        randmat_balance_cda.append(randmat_KNN_with_CDA_using_balanced_split(train_date_ = train_date_, test_date_ = test_date_, split = i))
    
    result[0, 5] = np.mean(pixmat_unbalanced_no_cda)
    result[0, 6] = np.std(pixmat_unbalanced_no_cda)
    result[1, 5] = np.mean(randmat_unbalanced_no_cda)
    result[1, 6] = np.std(randmat_unbalanced_no_cda)
    result[2, 5] = np.mean(pixmat_balance_no_cda)
    result[2, 6] = np.std(pixmat_balance_no_cda)
    result[3, 5] = np.mean(randmat_balance_no_cda)
    result[3, 6] = np.std(randmat_balance_no_cda)
    result[4, 5] = np.mean(pixmat_unbalance_cda)
    result[4, 6] = np.std(pixmat_unbalance_cda)
    result[5, 5] = np.mean(randmat_unbalance_cda)
    result[5, 6] = np.std(randmat_unbalance_cda)
    result[6, 5] = np.mean(pixmat_balance_cda)
    result[6, 6] = np.std(pixmat_balance_cda)
    result[7, 5] = np.mean(randmat_balance_cda)
    result[7, 6] = np.std(randmat_balance_cda)
    
    np.savetxt(train_date_ + "_" + test_date_ + "_linear_mapping_result.csv", result, fmt = "%d", delimiter = "\t", 
               header = "\ttrain_date\ttest_date\tbalanced_split\tapply_CDA\talg\tmean_accu\tstd_accu")

if __name__ == "__main__":
    main(train_date = "130411", test_date = "140416")