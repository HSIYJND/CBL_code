#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:37:02 2019

@author: yuanhang
"""
import sklearn
import numpy as np
from scipy.io import loadmat
from GLP import GLP
from LLP import LLP
from sklearn.metrics import balanced_accuracy_score

def demo_ver3(seed = 0, train_date = "130411", split = 0):
    data_path = "..data/"
    dataset = loadmat(data_path + "SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    # the split was made in matlab, index starts from 1 instead of 0
    train_indices = dataset["train_indices_splitter"][split] - 1 
    test_indices = dataset["test_indices_splitter"][split] - 1 

    data = dataset["spectra"][:, dataset["bbl"] == 1]
    # remove first two zero-bands
    data = data[:, 2:]
        
    labels = dataset["labels"] - 1
    labels = labels.astype(np.int)
    
    partial_labels = labels.copy()
    partial_labels[test_indices] = -1

    train_indices = np.where(partial_labels != -1)[0]
    test_indices = np.where(partial_labels == -1)[0]
    train_data, train_label = data[train_indices], labels[train_indices]
    test_data, test_label = data[test_indices], labels[test_indices]
    
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    knn.fit(train_data, train_label)
    predict = knn.predict(test_data)
    knn_accuracy = np.round((np.where(predict == test_label)[0].shape[0] / len(test_label)) * 100)
    print("knn,", knn_accuracy)
    
    glp = GLP(alpha = 0.99, n_neighbors = 30, method = "knn_adj")
    glp.train(train_data, train_label)
    predict = glp.test(test_data)
    glp_accuracy = np.round((np.where(predict == test_label)[0].shape[0] / len(test_label)) * 100)
    print("glp,", glp_accuracy)
    
    llp = LLP(alpha = 1, n_neighbors = 30, allow_negative = False)
    llp.train(train_data, train_label)
    predict = llp.test(test_data)
    llp_accuracy = np.round((np.where(predict == test_label)[0].shape[0] / len(test_label)) * 100)
    print("llp_accuracy,", llp_accuracy)


def GLPdemo(data_path = "..data/"):
    x = loadmat(data_path + "exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]
    
    # alg, alpha, propgation_weight, num_samples, unlabeled_rate, 
    # mean, std, weighted mean, weighted std
    # 2 * 2 * 3 * 19 = 228 lines
    
    for alpha in [0.99, 1]:
        for method in ["knn_adj", "knn_shortest", "knn_laplacian"]:
            result = np.zeros((19, 9)).astype(object)
            result[:, 0] = "GLP"
            result[:, 1] = "Hard_Clamping" if alpha == 1 else "Soft_Clamping" 
            result[:, 2] = method
            result[:, 3] = len(data)
            for unlabeled_rate in range(5, 100, 5):
                print("working...")
                row = unlabeled_rate // 5 - 1 
                result[row, 4] = unlabeled_rate
                temp = []
                temp_weighted = []
                # each method, run 10 times
                for seed in range(10):
                    rng = np.random.RandomState(seed)
                    unlabeled_percentage = unlabeled_rate/100
                    partial_labels = labels.copy()
                    for c in range(27):
                        c_idx = np.where(labels == c)[0]
                        while True:
                            unlabeled_idx = np.where((rng.rand(len(c_idx)) <= unlabeled_percentage) == True)[0]
                            if len(unlabeled_idx) != 0:
                                break
                        partial_labels[c_idx[unlabeled_idx]] = -1
                    glp = GLP(alpha = alpha, n_neighbors = 30, method = method)
                    train_indices = np.where(partial_labels != -1)[0]
                    test_indices = np.where(partial_labels == -1)[0]
                    train_data, train_label = data[train_indices], labels[train_indices]
                    test_data, test_label = data[test_indices], labels[test_indices]
                    glp.train(train_data, train_label)
                    predict = glp.test(test_data)
                    accuracy = np.round((np.where(predict == test_label)[0].shape[0] / len(test_label)) * 100)
                    print(accuracy)
                    temp.append(accuracy)
                    temp_weighted.append(balanced_accuracy_score(predict, test_label))
                result[row, 5] = np.mean(temp)
                result[row, 6] = np.std(temp)
                print("mean:", result[row, 5], "std:", result[row, 6])
            np.savetxt(result[0, 0] + "_" + result[0, 1] + "_" + method + "_result.csv", result, 
                       fmt='%s', delimiter='\t',header = "Alg\tClamping\tPropagation_Matrix\tNum_Samples\tUnlabeled_Rate\tAccuracy_Mean\tAccuracy_Std")
    
def LLPdemo(data_path = "..data/"):
    x = loadmat(data_path + "exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]
    for alpha in [0.99, 1]:
        for allow_negative in [False, True]:
            result = np.zeros((19, 7)).astype(object)
            result[:, 0] = "LLP"
            result[:, 1] = "Hard_Clamping" if alpha == 1 else "Soft_Clamping" 
            result[:, 2] = "allow_negative_weight" if not allow_negative else "nonnegative_weight_only"
            result[:, 3] = len(data)
            for unlabeled_rate in range(5, 100, 5):
                print("working...")
                row = unlabeled_rate // 5 - 1 
                result[row, 4] = unlabeled_rate
                temp = []
                # each method, run 10 times
                for seed in range(10):
                    rng = np.random.RandomState(seed)
                    unlabeled_percentage = unlabeled_rate/100
                    partial_labels = labels.copy()
                    for c in range(27):
                        c_idx = np.where(labels == c)[0]
                        while True:
                            unlabeled_idx = np.where((rng.rand(len(c_idx)) <= unlabeled_percentage) == True)[0]
                            if len(unlabeled_idx) != 0:
                                break
                        partial_labels[c_idx[unlabeled_idx]] = -1
                    llp = LLP(alpha = alpha, n_neighbors = 30, allow_negative = allow_negative)
                    train_indices = np.where(partial_labels != -1)[0]
                    test_indices = np.where(partial_labels == -1)[0]
                    train_data, train_label = data[train_indices], labels[train_indices]
                    test_data, test_label = data[test_indices], labels[test_indices]
                    llp.train(train_data, train_label)
                    predict = llp.test(test_data)
                    accuracy = np.round((np.where(predict == test_label)[0].shape[0] / len(test_label)) * 100)
                    temp.append(accuracy)
                result[row, 5] = np.mean(temp)
                result[row, 6] = np.std(temp)
                print("mean:", result[row, 5], "std:", result[row, 6])
            np.savetxt(result[0, 0] + "_" + result[0, 1] + "_" + result[0, 2] + "_result.csv", result, 
                       fmt='%s', delimiter='\t',header = "Alg\tClamping\tPropagation_Matrix\tNum_Samples\tUnlabeled_Rate\tAccuracy_Mean\tAccuracy_Std")

def KNNdemo(data_path = "..data/"):
    x = loadmat(data_path + "exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]
    result = np.zeros((19, 7)).astype(object)
    result[:, 0] = "KNN"
    for unlabeled_rate in range(5, 100, 5):
        print("working...")
        row = unlabeled_rate // 5 - 1 
        result[row, 4] = unlabeled_rate
        temp = []
        # each method, run 10 times
        for seed in range(10):
            rng = np.random.RandomState(seed)
            unlabeled_percentage = unlabeled_rate/100
            partial_labels = labels.copy()
            for c in range(27):
                c_idx = np.where(labels == c)[0]
                while True:
                    unlabeled_idx = np.where((rng.rand(len(c_idx)) <= unlabeled_percentage) == True)[0]
                    if len(unlabeled_idx) != 0:
                        break
                partial_labels[c_idx[unlabeled_idx]] = -1
            train_indices = np.where(partial_labels != -1)[0]
            test_indices = np.where(partial_labels == -1)[0]
            train_data, train_label = data[train_indices], labels[train_indices]
            test_data, test_label = data[test_indices], labels[test_indices]
            knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
            knn.fit(train_data, train_label)
            predict = knn.predict(test_data)
            accuracy = np.round((np.where(predict == test_label)[0].shape[0] / len(test_label)) * 100)
            temp.append(accuracy)
        result[row, 5] = np.mean(temp)
        result[row, 6] = np.std(temp)
        print("mean:", result[row, 5], "std:", result[row, 6])
    np.savetxt("KNN_result.csv", result, 
                       fmt='%s', delimiter='\t',header = "Alg\tClamping\tPropagation_Matrix\tNum_Samples\tUnlabeled_Rate\tAccuracy_Mean\tAccuracy_Std")
    
if __name__ == "__main__":
#    LLPdemo()
#    GLPdemo()
#    KNNdemo()
    demo_ver3()