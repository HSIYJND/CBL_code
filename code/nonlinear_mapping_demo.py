#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import sklearn.tree
import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nonlinear_mapping_model import NonlinearMappingPerceptron, NonlinearMappingMLP

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

# if __name__ == "__main__":
#     for i in range(5):
#         for train_date, test_date in [("130411", "140416"), ("140416", "130411")]:
#             for use_cda in [True, False]:
#                 for alg in ["pixmat", "randmat"]:
#                     for linear_init in [True, False]:
#                         for balance_split in [True, False]:
#                             demo(linear_init = linear_init, alg = alg, balance_split = balance_split, use_cda = use_cda, train_date = train_date, test_date = test_date, 
#                              activation = "modified_relu_tanh", split = i, hidden_unit = 0)