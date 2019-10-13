#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import sklearn.tree
import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from IEEE_paper_nonlinear_mapping_model import NonlinearMappingPerceptron, NonlinearMappingMLP

#%% BELOW ARE EXP WITHOUT CDA
def demo(linear_init = True, alg = "pixmat", balance_split = False, use_cda = False, train_date = "130411", test_date = "140416", 
         activation = "sigmoid", split = 0, hidden_unit = 0):
    
    data_path = "../data/"
    
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron(activation = activation, init_weight = None)
    else:
        # MLP
        net = NonlinearMappingMLP(activation = activation, init_weight = None)
    
    init_status = "linear_init" if linear_init else "rand_init"
    net_file_name = data_path + activation + "_" + init_status + "_" + alg + "_" + model_name + '_from_' + test_date + '_to_' + train_date + '.dat'
    net.load_state_from_file(net_file_name)
    
    if not balance_split:
        trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
        testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
        
        # the split was made in matlab, index starts from 1 instead of 0
        train_indices = trainset["train_indices_splitter"][split] - 1 
        
        # remove all bad bands
        train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
        # remove first two zero-bands
        train_spectra = train_spectra[train_indices, 2:].astype(np.float)
        train_targets = trainset["labels"][train_indices]
        
        # remove all bad bands
        test_spectra = testset["spectra"][:, testset["bbl"] == 1]
        # remove first two zero-bands
        test_spectra = test_spectra[:, 2:].astype(np.float)
        test_targets = testset["labels"]
        
    else:
        # balance_split is True
        trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
        testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
        
        # the split was made in python, index starts 0
        train_indices = trainset["train_indices_splitter"][split] 
        
        # bad bands and zero bands are moved when I was making the data file
        train_spectra = trainset["spectra"].astype(np.float)
        train_spectra = train_spectra[train_indices, :]
        train_targets = trainset["labels"][train_indices]
        
        # bad bands and zero bands are moved when I was making the data file
        test_spectra = testset["spectra"].astype(np.float)
        test_targets = testset["labels"]
    
    # nonlinear mapping
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()

    if use_cda:
        clf = LinearDiscriminantAnalysis(n_components = 26)
        clf.fit(train_spectra, train_targets)
        train_spectra = clf.transform(train_spectra)
        test_spectra = clf.transform(test_spectra)
    
    # use knn as a baseline    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    # make prediction, use the whole set
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    print("init:", linear_init, ",alg:", alg, ",balance_split:", balance_split, ",use_cda:",use_cda, 
          ",train:", train_date, ",test:",test_date, ",activation:", activation, 
          ",split:", split, ",accuracy:", accuracy)
    return accuracy

def nonlinear_pixmat_KNN_without_CDA_using_unbalanced_split(train_date = "130411", test_date = "140416",
                                                             activation = "sigmoid", split = 0, hidden_unit = 0):
    """
    EXP1 : non-pixmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    without CDA
    """
    
    data_path = "../data/"

    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron(activation = MyActivation(), init_weight = None)
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_pixmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')

    
    trainset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat(data_path + "SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    
    # the split was made in matlab, index starts from 1 instead of 0
    train_indices = trainset["train_indices_splitter"][split] - 1 
    
    # remove all bad bands
    train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    # remove first two zero-bands
    train_spectra = train_spectra[train_indices, 2:]
    train_targets = trainset["labels"][train_indices]
    
    # remove all bad bands
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    # remove first two zero-bands
    test_spectra = test_spectra[:, 2:]
    
    # map test date back to train date
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)

    # train knn
    KNN_classifier.fit(train_spectra, train_targets)
    
    # make prediction, use the whole set
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation, 
          "nonlinear_pixmat_KNN_without_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy
    
    
def nonlinear_randmat_KNN_without_CDA_using_unbalanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP2 : non-linear randmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    without CDA
    """
    
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
#        net = NonlinearMappingPerceptron()
        net = NonlinearMappingPerceptron(activation = MyActivation(), init_weight = None)
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_randmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation, 
          "nonlinear_randmat_KNN_without_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def nonlinear_pixmat_KNN_without_CDA_using_balanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP3 : nonlinear pixmat with KNN using Susan's split
    This function applies PixMat first, and then use KNN to classify.
    without CDA
    """
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron()
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_pixmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    test_spectra = testset["spectra"].astype(np.float)

    # map test date back to train date
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation, 
          "nonlinear_pixmat_KNN_without_CDA_balance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def nonlinear_randmat_KNN_without_CDA_using_balanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP4 : nonlinear randmat with KNN using Susan's split
    This function applies RandMat first, and then use KNN to classify.
    without CDA
    """
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron()
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_randmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    test_spectra = testset["spectra"].astype(np.float)

    # map test date back to train date
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation,
          "nonlinear_randmat_KNN_without_CDA_balance_split_pixel_level_accuracy:", accuracy)

    return accuracy

#%% BELOW ARE EXPs WITH CDA

def nonlinear_pixmat_KNN_with_CDA_using_unbalanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP5 : nonlinear pixmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    with CDA
    """
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron()
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_randmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    
    # apply CDA
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    # train knn on CDA features
    KNN_classifier.fit(train_spectra, train_targets)
    
    # remove all bad bands
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    # remove first two zero-bands
    test_spectra = test_spectra[:, 2:]
    
    # map test date back to train date
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction, use the whole set
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "acitvation:", activation, 
          "nonlinear_pixmat_KNN_with_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy
    
def nonlinear_randmat_KNN_with_CDA_using_unbalanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP6 : nonlinear randmat with KNN using Ron's split
    This function applies PixMat first, and then use KNN to classify.
    with CDA
    """
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron()
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_randmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation, 
          "nonlinear_randmat_KNN_with_CDA_unbalance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def nonlinear_pixmat_KNN_with_CDA_using_balanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP7 :nonlinear pixmat with KNN using Susan's split
    This function applies PixMat first, and then use KNN to classify.
    with CDA
    """
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron()
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_randmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    test_spectra = testset["spectra"].astype(np.float)

    # map test date back to train date
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation, 
          "nonlinear_pixmat_KNN_with_CDA_balance_split_pixel_level_accuracy:", accuracy)
    
    return accuracy

def nonlinear_randmat_KNN_with_CDA_using_balanced_split(train_date = "130411", test_date = "140416",
                                                  activation = "sigmoid", split = 0, 
                                                  hidden_unit = 0):
    """
    EXP8: nonlinear randmat with KNN using Susan's split
    This function applies RandMat first, and then use KNN to classify.
    with CDA
    """
    
    data_path = "../data/"
    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron()
    else:
        # MLP
        net = NonlinearMappingMLP(hidden_unit = hidden_unit)
    
    # load pre-trained parameters
    net.load_state_from_file(data_path + activation + '_randmat_' + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')
    
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
    test_spectra = testset["spectra"].astype(np.float)

    # map test date back to train date
    test_spectra = net(torch.FloatTensor(test_spectra)).cpu().detach().numpy()
    test_targets = testset["labels"]
    
    # apply CDA on test date
    test_spectra = clf.transform(test_spectra)
    
    # make prediction
    test_est = KNN_classifier.predict(test_spectra)
    
    accuracy = round(np.where(test_targets == test_est)[0].shape[0] / len(test_targets) * 100)
    
    print("split:", split, "train:", train_date, "test:", test_date, "activation:", activation,
          "nonlinear_randmat_KNN_with_CDA_balance_split_pixel_level_accuracy:", accuracy)
    
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
    
#    for i in range(5):
#        print("#### WITHOUT CDA ####################################################################################################")
#        pixmat_unbalanced_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_unbalanced_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        pixmat_balance_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_balance_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        print("#### WITH CDA ###########################################################################################################")
#        pixmat_unbalance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_unbalance_cda.append(nonlinear_randmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        pixmat_balance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_balance_cda.append(nonlinear_randmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
    
    for i in range(1):
        print("modified tanh!!!")
        print("#### WITHOUT CDA ####################################################################################################")
        pixmat_unbalanced_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_relu_tanh", split = i, hidden_unit = 0))
        randmat_unbalanced_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
        pixmat_balance_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
        randmat_balance_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
        print("#### WITH CDA ###########################################################################################################")
        pixmat_unbalance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
        randmat_unbalance_cda.append(nonlinear_randmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
        pixmat_balance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
        randmat_balance_cda.append(nonlinear_randmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "modified_tanh", split = i, hidden_unit = 0))
    
#    result[0, 5] = np.mean(pixmat_unbalanced_no_cda)
#    result[0, 6] = np.std(pixmat_unbalanced_no_cda)
#    result[1, 5] = np.mean(randmat_unbalanced_no_cda)
#    result[1, 6] = np.std(randmat_unbalanced_no_cda)
#    result[2, 5] = np.mean(pixmat_balance_no_cda)
#    result[2, 6] = np.std(pixmat_balance_no_cda)
#    result[3, 5] = np.mean(randmat_balance_no_cda)
#    result[3, 6] = np.std(randmat_balance_no_cda)
#    result[4, 5] = np.mean(pixmat_unbalance_cda)
#    result[4, 6] = np.std(pixmat_unbalance_cda)
#    result[5, 5] = np.mean(randmat_unbalance_cda)
#    result[5, 6] = np.std(randmat_unbalance_cda)
#    result[6, 5] = np.mean(pixmat_balance_cda)
#    result[6, 6] = np.std(pixmat_balance_cda)
#    result[7, 5] = np.mean(randmat_balance_cda)
#    result[7, 6] = np.std(randmat_balance_cda)
#    
#    np.savetxt(train_date_ + "_" + test_date_ + "_linear_mapping_result.csv", result, fmt = "%d", delimiter = "\t", 
#               header = "\ttrain_date\ttest_date\tbalanced_split\tapply_CDA\talg\tmean_accu\tstd_accu")
    
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
    
#    for i in range(5):
#        print("#### WITHOUT CDA ####################################################################################################")
#        pixmat_unbalanced_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_unbalanced_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        pixmat_balance_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_balance_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        print("#### WITH CDA ###########################################################################################################")
#        pixmat_unbalance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_unbalance_cda.append(nonlinear_randmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        pixmat_balance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
#        randmat_balance_cda.append(nonlinear_randmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
#                                                             activation = "sigmoid", split = i, hidden_unit = 0))
    
    for i in range(1):
        print("#### WITHOUT CDA ####################################################################################################")
        pixmat_unbalanced_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        randmat_unbalanced_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        pixmat_balance_no_cda.append(nonlinear_pixmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        randmat_balance_no_cda.append(nonlinear_randmat_KNN_without_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        print("#### WITH CDA ###########################################################################################################")
        pixmat_unbalance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        randmat_unbalance_cda.append(nonlinear_randmat_KNN_with_CDA_using_unbalanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        pixmat_balance_cda.append(nonlinear_pixmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
        randmat_balance_cda.append(nonlinear_randmat_KNN_with_CDA_using_balanced_split(train_date = train_date_, test_date = test_date_, 
                                                             activation = "tanh", split = i, hidden_unit = 0))
    
#    result[0, 5] = np.mean(pixmat_unbalanced_no_cda)
#    result[0, 6] = np.std(pixmat_unbalanced_no_cda)
#    result[1, 5] = np.mean(randmat_unbalanced_no_cda)
#    result[1, 6] = np.std(randmat_unbalanced_no_cda)
#    result[2, 5] = np.mean(pixmat_balance_no_cda)
#    result[2, 6] = np.std(pixmat_balance_no_cda)
#    result[3, 5] = np.mean(randmat_balance_no_cda)
#    result[3, 6] = np.std(randmat_balance_no_cda)
#    result[4, 5] = np.mean(pixmat_unbalance_cda)
#    result[4, 6] = np.std(pixmat_unbalance_cda)
#    result[5, 5] = np.mean(randmat_unbalance_cda)
#    result[5, 6] = np.std(randmat_unbalance_cda)
#    result[6, 5] = np.mean(pixmat_balance_cda)
#    result[6, 6] = np.std(pixmat_balance_cda)
#    result[7, 5] = np.mean(randmat_balance_cda)
#    result[7, 6] = np.std(randmat_balance_cda)
#    
#    np.savetxt(train_date_ + "_" + test_date_ + "_linear_mapping_result.csv", result, fmt = "%d", delimiter = "\t", 
#               header = "\ttrain_date\ttest_date\tbalanced_split\tapply_CDA\talg\tmean_accu\tstd_accu")

if __name__ == "__main__":
#    main(train_date = "130411", test_date = "140416")
#    for i in range(10):
#        demo(alg = "pixmat", balance_split = False, use_cda = False, train_date = "130411", test_date = "140416", 
#             activation = "modified_tanh", split = i, hidden_unit = 0)
#        demo(alg = "pixmat", balance_split = False, use_cda = False, train_date = "130411", test_date = "140416", 
#             activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#        demo(alg = "pixmat", balance_split = False, use_cda = True, train_date = "130411", test_date = "140416", 
#             activation = "modified_tanh", split = i, hidden_unit = 0)
#        demo(alg = "pixmat", balance_split = False, use_cda = True, train_date = "130411", test_date = "140416", 
#             activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#        
#        demo(alg = "pixmat", balance_split = True, use_cda = False, train_date = "130411", test_date = "140416", 
#             activation = "modified_tanh", split = i, hidden_unit = 0)
#        demo(alg = "pixmat", balance_split = True, use_cda = False, train_date = "130411", test_date = "140416", 
#             activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#        demo(alg = "pixmat", balance_split = True, use_cda = True, train_date = "130411", test_date = "140416", 
#             activation = "modified_tanh", split = i, hidden_unit = 0)
#        demo(alg = "pixmat", balance_split = True, use_cda = True, train_date = "130411", test_date = "140416", 
#             activation = "modified_relu_tanh", split = i, hidden_unit = 0)
        
#        for i in range(10):
#            demo(alg = "randmat", balance_split = False, use_cda = False, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "pixmat", balance_split = False, use_cda = False, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "randmat", balance_split = True, use_cda = False, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "pixmat", balance_split = True, use_cda = False, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "randmat", balance_split = False, use_cda = True, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "pixmat", balance_split = False, use_cda = True, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "randmat", balance_split = True, use_cda = True, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
#            
#            demo(alg = "pixmat", balance_split = True, use_cda = True, train_date = "140416", test_date = "130411", 
#                 activation = "modified_relu_tanh", split = i, hidden_unit = 0)
    for i in range(5):
        for train_date, test_date in [("130411", "140416"), ("140416", "130411")]:
            for use_cda in [True, False]:
                for alg in ["pixmat", "randmat"]:
                    for linear_init in [True, False]:
                        for balance_split in [True, False]:
                            demo(linear_init = linear_init, alg = alg, balance_split = balance_split, use_cda = use_cda, train_date = train_date, test_date = test_date, 
                             activation = "modified_relu_tanh", split = i, hidden_unit = 0)