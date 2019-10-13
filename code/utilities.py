#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy.io import savemat, loadmat
from itertools import permutations, combinations
from glob import glob
import time
import pandas as pd
from hausdorff import hausdorff
from geopy.distance import geodesic
from math import sqrt
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import warnings
import torch.utils.data as tdata
import sklearn
import sklearn.metrics
import scipy.signal

def pixmat(train_spectra, train_polygons, train_coordinates,
           test_spectra, test_polygons, test_coordinates):
    """
    This function performs pixel-matching linear mapping.
    Only the pixels that are included in both two dates are used for constructing pairs.
    
    Inputs:
    train_spectra       : N_train x 174 Numpy array, spectra of pixels of training date
                          N_train : number of pixels of training date
                          174 : number of good bands (waterbands and zero bands are removed)
    train_polygon_names : N_train x 1 Numpy array, name of the polygon that a pixel belongs to 
    train_coordinates   : N_train x 2 Numpy array, relative coordinates (X,Y) of pixels in the HSI
    test_spectra        : N_test x 174 Numpy array, spectra of pixels of testing date
                          N_test : number of pixels of testing date 
                          174 : number of good bands (waterbands and zero bands are removed)
    test_polygon_names  : N_test x 1 Numpy array, name of the polygon that a pixel belongs to 
    test_coordinates    : N_test x 2 Numpy array, relative coordinates (X,Y) of pixels in the HSI
    
    Outputs:
    A                   : 174 x 174 Numpy array, linear mapping matrix that maps spectra of training date
                          to spectra of testing date
    """
    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []

    # Step 2 : iterate all common polygons and construct pairs by checking relative coordinates (X, Y) of the pixels
    for common_polygon in common_polygons:
        train_indices = np.where(train_polygons == common_polygon)[0]
        test_indices = np.where(test_polygons == common_polygon)[0]
        train_pos = train_coordinates[train_indices, :].tolist()
        test_pos = test_coordinates[test_indices, :].tolist()
        # For each pixel, if it is included in both training and testing date, use the spectra of two dates to construct pair
        # Otherwise, discard it, i.e., not all training pixels can be used to construct pairs
        for i in range(len(test_pos)):
            try:
                idx = train_pos.index(test_pos[i])
                X_train.append(train_spectra[train_indices[idx], :])
                X_test.append(test_spectra[test_indices[i], :])
            except ValueError:
                continue
    X_train = np.vstack(X_train).astype('float')
    X_test = np.vstack(X_test).astype('float')
    A = np.linalg.lstsq(X_test, X_train)[0] 
    return A

def randmat(train_spectra, train_polygons, test_spectra, test_polygons):
    """
    This function performs random-matching linear mapping by shuffling.
    One pixel can be used at most once to construct pairs.
    
    Inputs:
    train_spectra       : N_train x 174 Numpy array, spectra of pixels of training date
                          N_train : number of pixels of training date
                          174 : number of good bands(excluding waterbands and zero bands)
    train_polygon_names : N_train x 1 Numpy array, name of the polygon that a pixel belongs to 
    test_spectra        : N_test x 174 Numpy array, spectra of pixels of testing date
                          N_test : number of pixels of testing date 
    test_polygon_names  : N_test x 1 Numpy array, name of the polygon that a pixel belongs to 
    
    Outputs:
    A                   : 174 x 174 Numpy array, linear mapping matrix that maps spectra of training date
                          to spectra of testing date
    """
    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []
     # Step 2 : iterate all the common polygons and construct pairs by random shuffling
    for common_polygon in common_polygons:
        train_indices = np.where(train_polygons == common_polygon)[0]
        test_indices = np.where(test_polygons == common_polygon)[0]
        train_pixels = train_spectra[train_indices, :]
        test_pixels = test_spectra[test_indices, :]
        # Not all training pixels are guaranteed to be used for constructing pairs 
        min_length = min(len(train_pixels), len(test_pixels))
        perm_train = np.random.permutation(len(train_indices))
        perm_train = perm_train[:min_length]
        train_pixels = train_pixels[perm_train, :]
        perm_test = np.random.permutation(len(test_indices))
        perm_test = perm_test[:min_length]
        test_pixels = test_pixels[perm_test, :]
        X_train.append(train_pixels)
        X_test.append(test_pixels)
    X_train = np.vstack(X_train).astype('float')
    X_test = np.vstack(X_test).astype('float')
    A = np.linalg.lstsq(X_test, X_train)[0] 
    return A

def randmat_ver2(train_spectra, train_polygons, test_spectra, test_polygons):
    """
    This function performs random-matching linear mapping by randomly selection.
    One pixel can be used for MULTIPLE times to construct pairs.
    
    Inputs:
    train_spectra       : N_train x 174 Numpy array, spectra of pixels of training date
                          N_train : number of pixels of training date
                          174 : number of good bands(excluding waterbands and zero bands)
    train_polygon_names : N_train x 1 Numpy array, name of the polygon that a pixel belongs to 
    test_spectra        : N_test x 174 Numpy array, spectra of pixels of testing date
                          N_test : number of pixels of testing date 
    test_polygon_names  : N_test x 1 Numpy array, name of the polygon that a pixel belongs to 
    
    Outputs:
    A                   : 174 x 174 Numpy array, linear mapping matrix that maps spectra of training date
                          to spectra of testing date
    """
    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []
     # Step 2 : iterate all the common polygons and construct pairs by randomly selection
    for common_polygon in common_polygons:
        train_indices = np.where(train_polygons == common_polygon)[0]
        test_indices = np.where(test_polygons == common_polygon)[0]
        train_pixels = train_spectra[train_indices, :]
        test_pixels = test_spectra[test_indices, :]
        for i in range(len(test_pixels)):
            # Note: a training pixel may be used multiple times to construct pairs
            j = np.random.randint(0, len(train_pixels))
            X_train.append(train_pixels[j, :])
            X_test.append(test_pixels[i, :])
    X_train = np.vstack(X_train).astype('float')
    X_test = np.vstack(X_test).astype('float')
    A = np.linalg.lstsq(X_test, X_train)[0] 
    return A

def get_date(file_name):
    """
    Input:
    file_name     : Python built-in string, raw data csv file
                    Below are the legal inputs:
                    *_AVIRIS_speclib_subset_spectra.csv
                    *_AVIRIS_speclib_subset_metadata.csv
                    *_AVIRIS_speclib_subset_trainval.csv
                    where * is the date

    Output:
    date          : Python built-in string
    """
    date = file_name.split('_')[0]
    return date

def get_auxiliary_info(file_name):
    """
    Input:exp1()
    file_name          : Python built-in string
                         Must be auxiliary_info.mat
    
    Output:
    bbl                : 224 x 1 Numpy Array, 0 indices good band, 1 indices bad band
    label_name_mapping : Python built-in dictionary, (key, value) : (class_name, label)
    """
    raw_data = loadmat(file_name, squeeze_me = True)
    bbl = raw_data["bbl"]
    label_name_mapping = {name:int(i) for i, name in enumerate(raw_data["label_names"])}
    return bbl, label_name_mapping
    

def get_spectra_and_polygon_name(file_name, bbl):
    """
    Inputs:
    file_name     : Python built-in string, raw data csv file
                    Must be *_AVIRIS_speclib_subset_spectra.csv, where * is date
    bbl           : 224 x 1 Numpy array

    Outputs:
    spectra       : Npixel x 174 Numpy array
    polygon_names : Npixel x 1 Numpy array
    """
    data = pd.read_csv(file_name)
    polygon_names = data['PolygonName'].values
    spectra = data.values[:,5:]
    spectra = spectra[:, bbl == 1] # remove water bands, 224 to 176
    spectra = spectra/10000
    spectra = spectra[:, 2:] # first two bands are zero bands, 176 to 174
    spectra[spectra < 0] = 0 
    spectra[spectra > 1] = 1 
    return spectra, polygon_names

def get_thermal(file_name, unique_polygons):
    """
    Inputs:
    file_name       : Python built-in string, raw data csv file
                      Must be *_AVIRIS+MASTER_speclib_spectra.csv, where * is date
    unique_polygons : Npolygon x 1 Numpy array

    Outputs:
    polygon_thermal : Npolygon x 5 Numpy array
                      each row corresponds to each unique_polygon in unique_polygons
    """
    data = pd.read_csv(file_name)
    pixel_thermal = data.values[:, -10:-5]/10000
    pixel_polygon = data.values[:, 2]
    polygon_thermal = np.zeros((unique_polygons.shape[0], 5))
    for i, unique_polygon in enumerate(unique_polygons):
        indices = np.where(pixel_polygon == unique_polygon)[0]
        if len(indices) == 0:
            continue
        else:
            polygon_thermal[i, :] = polygon_aggregation(pixel_thermal[indices, :], aggregate_using_truncated_mean)     
    return polygon_thermal

def get_gis_feature(file_name, unique_polygons):
    """
    Inputs:
    file_name       : Python built-in string, raw data csv file
                      Must be *_AVIRIS+MASTER_speclib_spectra.csv, where * is date
    unique_polygons : Npolygon x 1 Numpy array

    Outputs:
    polygon_gis     : Npolygon x 7 Numpy array
                      each row corresponds to each unique_polygon in unique_polygons
    """
    data = pd.read_csv(file_name)
    pixel_gis = data.values[:, 5:]
    polygon_gis = np.zeros((unique_polygons.shape[0], pixel_gis.shape[1]))
    pixel_polygon_name = data.values[:, 2]
    pixel_gis = (pixel_gis - np.min(pixel_gis, axis = 0)) / (np.max(pixel_gis, axis = 0) - np.min(pixel_gis, axis = 0))
    for i, unique_polygon in enumerate(unique_polygons):
        indices = np.where(pixel_polygon_name == unique_polygon)[0]
        if len(indices) == 0:
            continue
        else:
            polygon_gis[i, :] = polygon_aggregation(pixel_gis[indices, :], aggregate_using_truncated_mean)     
    return polygon_gis
    
def get_spectra(file_name, bbl, unique_polygons):
    """
    Inputs:
    file_name       : Python built-in string, raw data csv file
                      Must be *_AVIRIS_speclib_subset_spectra.csv, where * is date
    bbl             : 224 x 1 Numpy array
    unique_polygons : Npolygon x 1 Numpy array

    Outputs:
    polygon_spectra : Npixel x 174 Numpy array
    """
    pixel_spectra, polygon_names = get_spectra_and_polygon_name(file_name, bbl)
    data = pd.read_csv(file_name)
    pixel_polygon = data['PolygonName'].values
    pixel_spectra = data.values[:,5:]
    pixel_spectra = pixel_spectra[:, bbl == 1] # remove water bands, 224 to 176
    pixel_spectra /= 10000
    pixel_spectra = pixel_spectra[:, 2:] # first two bands are zero bands, 176 to 174
    pixel_spectra[pixel_spectra < 0] = 0 
    pixel_spectra[pixel_spectra > 1] = 1 
    polygon_spectra = np.zeros((unique_polygons.shape[0], 174))
    for i, unique_polygon in enumerate(unique_polygons):
        indices = np.where(pixel_polygon == unique_polygon)[0]
        if len(indices) == 0:
            continue
        else:
            polygon_spectra[i, :] = polygon_aggregation(pixel_spectra[indices, :], aggregate_using_truncated_mean)   
            print(i, "pause for debugging")
            print(polygon_spectra[i, :])
    return polygon_spectra

def assign_labels_for_polygons(label_name_mapping, unique_polygons):
    """
    Inputs:
    label_name_mapping    : Python built-in dictionary, (key, value): (class_name, label)
    unique_polygons       : Npolygon x 1 Numpy array, e.g., QUDO_004, (class_name)_number
    
    Outputs:
    polygon_labels        : Npolygon x 1 Numpy array
    """
    # label_name_mapping = {name:i for i, name in enumerate(auxiliary_info["label_names"])}
    polygon_labels = -1*np.ones(unique_polygons.shape)
    for i in range(len(unique_polygons)):
        polygon_labels[i] = label_name_mapping[(unique_polygons[i][0].split('_')[0])]
    polygon_labels = polygon_labels.astype(np.int)
    return polygon_labels

def get_coordinates(file_name):
    """
    Inputs:
    file_name     : Python built-in string, raw data csv file
                    Must be *_AVIRIS_speclib_subset_metadata.csv, where * is date

    Outputs:
    coordinates   : Npixel x 2 Numpy array
    """
    coordinates = pd.read_csv(file_name)[['X','Y']].values
    return coordinates

def split_dataset(file_name, polygon_names, index):
    """
    Split by polygon, without aggreagation
    
    Inputs:
    file_name     : Python built-in string, raw data csv file
                    Must be *_AVIRIS_speclib_subset_trainval.csv, where * is date
    polygon_names : Npixel x 1 Numpy array
    index         : Python built-in integer in range [0, 49]

    Outputs:
    split         : Npixel x 1 Numpy array
                    0 for testing, 1 for training, 2 for validation
                    Each element corresponds to the input polygon_names 
    """
    assert(type(index) is int and index >= 0 and index <= 49)
    split = pd.read_csv(file_name).values
    split = split[:, index]
    train_pixels_indices = []
    valid_pixel_indices = []
    test_pixels_indices = []
    unique_polygons = np.unique(polygon_names)
    unique_polygons.astype(str)
    for unique_polygon in unique_polygons:
        temp_indices = np.where(polygon_names == unique_polygon)[0]
        if np.all(split[temp_indices] == 0):
            # if all pixels of a polygon are labeled 0, this polygon is a testing polygon
            # all of its pixels are used for testing
            test_pixels_indices += temp_indices.tolist()
        elif np.all(split[temp_indices] == 1):
            # if all pixels of a polygon are labeled 1, this polygon is a training polygon
            # if this polygon has no more than 10 pixels, all of its pixels are used for training
            train_pixels_indices += temp_indices.tolist()
        else:
            # if a polygon is selected as training polygon but has more than 10 pixels
            # all the pixels labeled 0 are used for VALIDATION, they are going to be relabeled as 2 
            # while the pixels labeled 1 are used for TRAINING
            train_pixels_indices += temp_indices[np.where(split[temp_indices] == 1)[0]].tolist()
            valid_pixel_indices += temp_indices[np.where(split[temp_indices] == 0)[0]].tolist()
        split[train_pixels_indices] = 1 # 1 for training
        split[valid_pixel_indices] = 2 # 2 for validation
        split[test_pixels_indices] = 0 # 0 for testing
    split = split.astype(np.int)
    return split

def get_cda_transformation_matrix(spectra, split, labels, reduced_dim):
    """
    Inputs:
    spectra       : Npixel x 174 Numpy array
    split         : Npixel x 1 Numpy array
                    0 for testing, 1 for training, 2 for validation
    labels        : Npixel x 1 Numpy array
    reduced_dim   : Python integer, at most (Nclass - 1)
                    Nclass: number of different classes

    Output:
    A             : 174 x reduced_dim Numpy array
    """
    clf = LinearDiscriminantAnalysis(n_components = reduced_dim)
    indices = split[np.where(split == 1)[0]]
    spectra = spectra[indices, :]
    labels = labels[indices, :].astype(np.int64)
    clf.fit(spectra, labels)
    return clf.scalings_

def get_all_cda_transformation_matrices(aux_file, spectra_file, split_file):
    """
    Inputs:
    spectra       : Npixel x 174 Numpy array
    split         : Npixel x 1 Numpy array
                    0 for testing, 1 for training, 2 for validation
    labels        : Npixel x 1 Numpy array
    reduced_dim   : Python integer, at most (Nclass - 1)
                    Nclass: number of different classes

    Output:
    A             : 174 x reduced_dim Numpy array
    """
    bbl, label_name_mapping = get_auxiliary_info(aux_file)

    clf = LinearDiscriminantAnalysis(n_components = reduced_dim)
    indices = split[np.where(split == 1)[0]]
    spectra = spectra[indices, :]
    labels = labels[indices, :].astype(np.int64)
    clf.fit(spectra, labels)

    return clf.scalings_

def split_dataset_for_aggregated_polygon(file_name, polygon_names, unique_polygons, index):
    """
    Inputs:
    file_name       : Python built-in string, raw data csv file
                      Must be *_AVIRIS_speclib_subset_trainval.csv, where * is date
    polygon_names   : Npixel x 1 Numpy array
    index           : Python built-in integer in range [0, 49]

    Outputs:
    polygon_split   : Npolygon x 1 Numpy array
                      0 for testing, 1 for training, 2 for validation
    unique_polygons : Npolygon x 1 Numpy array, corresponding to polygon_split
    """
    assert(type(index) is int and index >= 0 and index <= 49)
    split = pd.read_csv(file_name).values
    split = split[:, index]
    polygon_split = -1*np.ones(unique_polygons.shape)

    for i, unique_polygon in enumerate(unique_polygons):
        temp_indices = np.where(polygon_names == unique_polygon)[0]
        if np.all(split[temp_indices] == 0):
            # if all pixels of a polygon are labeled 0, this polygon is a testing polygon
            # all of its pixels are used for testing
            polygon_split[i] = 0
        elif np.all(split[temp_indices] == 1):
            # if all pixels of a polygon are labeled 1, this polygon is a training polygon
            # if this polygon has no more than 10 pixels, all of its pixels are used for training
            polygon_split[i] = 1
        else:
            # if a polygon is selected as training polygon but has more than 10 pixels
            # all the pixels labeled 0 are used for VALIDATION, they are going to be relabeled as 2 
            # while the pixels labeled 1 are used for TRAINING
            polygon_split[i] = 2
    
    polygon_split = polygon_split.astype(np.int)
    return polygon_split


def graph_fourier_transform(A, normalized = "symmetric"):
    """
    This function applies Graph Convolution on a graph represented by adjacency matrix A
    by eigen-decomposition. This function takes adjacency matrix as input, and returns
    the eigenvectors and eigenvalues of the Laplacian matrix, and a Python dictionary 
    contains number of connected components and the conditional number of the Laplacian matrix.  
    
    Inputs:
    A         : V x V Numpy array, adjacency matrix
                V: number of vertices in the graph, 
    normalized: Python built-in string, normalization parameter
                "unnormalized" means without normalization
                "symmetric" means symmetric normalized
                "random" means random walk walk normalized
    
    Outputs:
    U         : V x V Numpy array, eigenvectors of the Laplacian matrix
    S         : V x 1 Numpy array, eigenvalues of the Laplacian matrix
    info      : Python built-in dictionary which contains number of connected components 
                and the condition number of the Laplacian matrix

    Reference:
    Eigendecomposition     :   https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
    Laplacian matrix       :   https://en.wikipedia.org/wiki/Laplacian_matrix
    Graph fourier transform:   https://users.ece.cmu.edu/~asandryh/papers/icassp13a.pdf
    Positive semi-definite :   https://en.wikipedia.org/wiki/Definiteness_of_a_matrix
    """

    # sum the adjacency matrix by row to get the degree matrix 
    D = np.diag(np.sum(A, axis = 1)) 
    if normalized == "unnormalized":
        L = D - A
    elif normalized == "symmetric":
        D_inv_sqrt = np.sqrt(np.linalg.inv(D))
        I = np.eye(A.shape[0])
        L = I - np.linalg.multi_dot([D_inv_sqrt, A, D_inv_sqrt])
    elif normalized == "random":
        I = np.eye(A.shape[0])
        L = I - np.dot(np.linalg.inv(D), A)
    else:
        raise ValueError("Sorry. Unrecognized Normalization Type")
    
    if not np.all(np.linalg.eigvals(L) >= 0):
        # if L is not positive semi-definite
        # we apply diagonal loading to make it PSD
        warnings.warn("Laplacian matrix is not PSD, we apply Diagonal Loading to make it PSD")
        print("Before Diagonal Loading, the condition number of L is:", np.linalg.cond(L))
        L += 0.001*np.trace(L)*np.eye(L.shape[0])
        print("After Diagonal Loading, the condition number of L is:", np.linalg.cond(L))

    S, U = np.linalg.eig(L)

    # numpy sorts eigenvalues ascending by default
    # we want the eigenvalues sorted in descending order
    
    U = np.fliplr(U)
    U = np.real(U)
    S.sort()
    S = S[::-1]
    
    num_components = A.shape[0] - np.linalg.matrix_rank(L)
    condition_number = np.linalg.cond(L)
    info = {"num_components": num_components, 
            "condition_number":condition_number}
    
    return U, S, info

def polygon_aggregation(data, aggregation_method = None):
    """
    Call once, aggregates ONE polygon!

    This function aggregates multi-source data of pixels insides a polygon. 
    All pixels inside a polygon share same label, polygon_name, but have different data obtained by multiple sensors. 
    This function aggregates all the pixels that belong to the same polygon using aggregation_method.
    By aggregation, various-size polygons can be represented using several fixed-size feature vectors.
    
    Inputs:
    data               : (1) Single-sensor data: Npixel x D Numpy array, Npixel: number of pixels of a polygon 
                             D: number of features obtained by sensor
                         (2) Multi-sensor data: Python built-in list of N (Npixel x Di) Numpy arrays
                             N: number of sensors, Npixel: number of pixels of a polygon
                             Di: number of features obtained by sensor i, i = 1, 2, ... N
                         
    aggregation_method : Python function, the aggregation method which takes sensor data, i.e., 
                         N x 1 Numpy array of (Npixel x Di) Numpy arrays as input and returns aggregated data of the polygon
                         N x 1 Numpy array of (1 x Di) Numpy arrays
    
    Outputs:
    polygon_data       : (1) Single-sensor data: 1 x D Numpy array, D: number of features obtained by sensor
                         (2) Multi-sensor data: Python list contains N (1 x Di) Numpy arrays, aggregated data of each polygon
                             N: number of sensors, Di: number of features obtained by sensor i, i = 1, 2, ... N

    Reference:
    Different kinds of mean:   https://en.wikipedia.org/wiki/Mean
    """
    
    if type(data) is list:
        num_sensors = len(data)
    elif type(data) is np.ndarray:
        num_sensors = 1
    else:
        raise ValueError("Sorry. Unrecognized data type")
    data = data.astype(np.float)
    polygon_data = [] if num_sensors > 1 else None
    for i in range(num_sensors):
        sensor_data = data[i] if num_sensors > 1 else data[:]
        sensor_data = aggregation_method(sensor_data)
        if num_sensors > 1:
            polygon_data.append(sensor_data)
        else: 
            polygon_data = sensor_data
    return polygon_data

def aggregate_using_unweighted_mean(sensor_data):
    """
    This function takes sensor data of pixels as input, aggregates by unweighted average,
    and returns aggregated sensor data of the polygon.sensor_data
    
    Inputs:
    sensor_data:  Npixel x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 

    Outputs:
    sensor_data:  1 x D Numpy arrays, D: number of features 

    """
    sensor_data = np.mean(sensor_data, axis = 0)
    sensor_data = sensor_data.reshape((1, sensor_data.shape[0]))
    return sensor_data

def aggregate_using_weighted_mean(sensor_data, ord = None):
    """
    This function takes sensor data of pixels as input, aggregates by weighted average(the weights 
    are the L2 norm of sensor data of each pixel), and returns aggregated sensor data of the polygon.
    
    Inputs:
    sensor_data:  Npixel x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 
    ord        :  norm of matrices 
                  "None" means L2-norm
                  For more infomation please refer to numpy docs
                  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
                  
    Outputs:
    sensor_data:  1 x D Numpy arrays, D: number of features 

    Reference:
    Weighted arithmetic mean : https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """

    weights = np.linalg.norm(sensor_data, ord = ord, axis = 1)
    weights = weights/sum(weights)
    weights = weights.reshape((sensor_data.shape[0],1))
    sensor_data = weights.T@sensor_data
    return sensor_data

def aggregate_using_truncated_mean(sensor_data, ord = None, alpha = 0.1):
    """
    This function takes sensor data of pixels as input, aggregates by weighted average(the weights 
    are the L2 norm of sensor data of each pixel), and returns aggregated sensor data of the polygon.
    
    Inputs:
    sensor_data:  Npixel x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 
    ord        :  norm of matrices 
                  "None" means L2-norm
                  For more infomation please refer to numpy docs
                  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    alpha      :  float, alpha-trimmed percentage used for truncated mean, in the range of [0, 0.5)

    Outputs:
    sensor_data:  1 x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 

    Reference:
    Truncated mean : https://en.wikipedia.org/wiki/Truncated_mean

    """
    weights = np.linalg.norm(sensor_data, axis = 1)
    order = np.argsort(weights)
    start = int(len(order)*alpha)
    end = len(order)-int(len(order)*alpha)
    # we want to keep order[start:end], where end is not included
    if len(order) - 2*int(len(order)*alpha) < 1: 
        order = order[1:-1] if len(order) >= 3 else order
    else:
        order = order[int(len(order)*alpha):len(order)-int(len(order)*alpha)]
    sensor_data = np.mean(sensor_data[order,:], axis = 0)
    sensor_data = sensor_data.reshape(1, sensor_data.shape[0])
    return sensor_data

def construct_adjacency_matrix(dist, method = "fixed", n_neighbors = 10, 
                 sigma = 1.0, normalized = True, sparse = False):
    """
    This function takes distance matrix(dissimilarity matrix) as inputs and 
    returns adjacency matrix based on the distance matrix using KNN
    (fixed-size number KNN / radius-based KNN).
    
    Inputs:
    dist        : V x V Numpy array, distance matrix(dissimilarity matrix)
    method      : Python string, KNN implementation method
                 "radius" means constructing A by using radius-based KNN
                 "fixed"  means constructing A by using fixed-number KNN
    radius      : float value, radius used for RBF kernel
    n_neighbors : int value, number of nearest neighbors for fixed-number KNN
    sigma       : float value, magic number used for RBF kernel
    sparse      : bool, true means convert adjacency matrix to sparse matrix
    
    Outputs:pixel_polygon
    adj          : V x V (Sparse) Numpy array, adjacency matrix

    Reference:
    K nearest neighbors algorithm: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    """
    adj = np.zeros(dist.shape)
    if method == "radius":
        adj = np.exp(-(dist - dist.T)**2 / sigma**2)
        adj[np.isnan(adj)] = 0
    elif method == "fixed":
        for row in range(dist.shape[0]):
            temp = np.argsort(dist[row,:])[:n_neighbors] #indices
            if normalized:
                weights = dist[row, temp] / dist[row,temp[-1]] # in range[0,1]
                weights = np.exp(-weights/0.5)
                for i in range(len(temp)): 
                    adj[row, temp[i]] = weights[i]
    else:
        # currently we only support 2 ways of KNN
        raise ValueError("Sorry. Unrecognized KNN construction method")
    if sparse:
        adj = scipy.sparse.csr_matrix(adj)
    return adj
  
def construct_distance_matrix(data, polygon_names, metric = "Haudsdroff", norm = "L1", 
                 sensor = "AVIRIS"):
    """
    This function regards polygons as bags of several pixels and calculates pairwise Haudsdroff distances
    between polygons based on different metrics(e.g, L1/L2).
    
    Inputs:
    data          : Nsamples x D Numpy array, D: number of features
    polygon_names : Nsamples x 1 Numpy array
    Metric        : Python string, metric used for measuring pairwise distances
                    "Haudsdroff" means using Haudsdroff distance
    norm          : Python string
                    "L1": manhatthan distance
                    "L2": Euclidean distance
    sensor        : Python string, sensor type
                    "AVIRIS": hyperspectral data
                    "MASTER": thermal infrared data
                    "GIS"   : GIS data

    Outputs:
    dist          : Nsamples x Nsamples Numpy array, distance matrix

    Reference:
    L1, L2 norm           :  https://en.wikipedia.org/wiki/Norm_(mathematics)
    Haudsdroff distance   :  https://en.wikipedia.org/wiki/Hausdorff_distance
    """
    idx_mapping = {}
    polygons = np.array([]).astype(object)
    
    for polygon_name in polygon_names: 
        if polygon_name not in idx_mapping:
            idx_mapping[polygon_name] = np.where(polygon_name == polygon_names)[0]
            polygons = np.append(polygons, np.array(polygon_name))
            
    # for missing data, we regard the distance as inf, thus, they are not connected in the adjacency matrix
    dist = np.zeros((len(polygons), len(polygons)))*float('inf')
    
    if sensor == "AVIRIS" or sensor == "MASTER":
        if metric == "Haudsdroff":
            for pair in combinations(range(len(polygons)), 2):
                dist[pair[0], pair[1]] = hausdorff(data[idx_mapping[polygons[pair[0]]],:], 
                                                   data[idx_mapping[polygons[pair[1]]],:], distance = norm)
                dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
        else:
            # currently we only use Haudsdroff distance --- multiple instance learning
            raise ValueError("Sorry. Unrecognized metric type")
    elif sensor == "GIS":
        # the first three features obtained by GIS are longtitde, latitude and elevation 
        long_lat, elevation = data[:,0:2], data[:, 2]
        for pair in combinations(range(len(polygons)), 2):
            # convert geodesic distance to euclidean distance
            long_lat_distance = geodesic(long_lat[idx_mapping[polygons[pair[0]]]], long_lat[idx_mapping[polygons[pair[1]]]]).km
            elevation_distance = abs(elevation[pair[0]] - elevation[pair[1]]) / 1000 # in km
            dist[pair[0], pair[1]] = sqrt(long_lat_distance**2 + elevation_distance**2) 
            dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
    else:
        raise ValueError("Sorry. Unrecognized data sensor")
    
    return dist

def construct_distance_matrix_using_aggregated_polygon(data):
    """
    This function measures distance between aggregated polygon using L1 norm.
    
    Input:
    data          : Nsamples x D Numpy array, D: number of features

    Output:
    dist          : Nsamples x Nsamples Numpy array, distance matrix

    Reference:
    L1, L2 norm           :  https://en.wikipedia.org/wiki/Norm_(mathematics)
    """
    num_polygons = len(data)
    dist = np.ones((num_polygons, num_polygons))*0.001
    for pair in combinations(range(num_polygons), 2):
        dist[pair[0], pair[1]] = sklearn.metrics.pairwise.manhattan_distances(data[pair[0], :].reshape(1,-1), 
                                                                              data[pair[1], :].reshape(1,-1))
        dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
    return dist

def get_statistic(data):
    """
    This function takes data of sensor(s) as input, drops the missing data (0s or nan)
    and returns the mean of the features obtained by the sensor(s).
    
    Input:
    data              : (1) Single-sensor : Npolygon x D Numpy array, D is the number of features
                        (2) Multi-sensor  : Python built-in list of N (Npixel x Di) Numpy arrays
                            N: number of sensors,  Di: number of features obtained by sensor i, i = 1, 2, ... N
          
    Output:
    feature_mean      : (1) Single-sensor : 1 x D Numpy array
                        (2) Multi-sensor  : Python built-in list of N (1 x Di) Numpy arrays
    missing_data_flag : (1) Single-sensor : Npolygon x 1 Numpy array, True indicates missing data
                        (2) Multi-sensor  : Python built-in list of N (Npolygon x 1) Numpy arrays
                            N: number of sensors
    """
    if type(data) is list:
        num_sensors = len(data)
        feature_mean = []
        missing_data_flag = []
        for i in range(num_sensors):
            flag = np.logical_or(np.all(data[i] == 0, axis = 1), np.all(np.isnan(data[i]), axis = 1))[:, np.newaxis]
            print(flag)
            temp = data[i][~np.squeeze(flag), :]
            feature_mean.append(np.mean(temp, axis = 0).reshape(1, temp.shape[1]))
            missing_data_flag.append(flag)
    elif type(data) is np.ndarray:
        ### missing_dada_flag : True means missing data
        missing_data_flag = np.logical_or(np.all(data == 0, axis = 1), np.all(np.isnan(data), axis = 1))[:, np.newaxis]
        temp = data[~np.logical_or(np.all(data == 0, axis = 1), np.all(np.isnan(data), axis = 1)), :]
        feature_mean = np.mean(temp, axis = 0).reshape(1, data.shape[1])
    else:
        raise ValueError("Sorry. Unrecognized data type")
    return feature_mean, missing_data_flag

def replace_missing_data(data, feature_mean, missing_data_flag):
    """
    This function replaces missing date with mean of features accuired by sensors.
    
    Input:
    data              : (1) Single-sensor : Npolygon x D Numpy array, D is the number of features
                        (2) Multi-sensor  : Python built-in list of N (Npixel x Di) Numpy arrays
                            N: number of sensors,  Di: number of features obtained by sensor i, i = 1, 2, ... N
    feature_mean      : (1) Single-sensor : 1 x D Numpy array
                        (2) Multi-sensor  : Python built-in list of N (1 x Di) Numpy arrays
    missing_data_flag : (1) Single-sensor : Npolygon x 1 Numpy array, True indicates missing data
                        (2) Multi-sensor  : Python built-in list of N (Npolygon x 1) Numpy arrays
                            N: number of sensors
    
    Output:
    data              : replaced 0/nan with sensor means
                        (1) Single-sensor : Npolygon x D Numpy array, D is the number of features
                        (2) Multi-sensor  : Python built-in list of N (Npixel x Di) Numpy arrays
                            N: number of sensors,  Di: number of features obtained by sensor i, i = 1, 2, ... N
    """
    if type(data) is list:
        num_sensors = len(data)
        for i in range(num_sensors):
            data[i][np.squeeze(missing_data_flag)] = feature_mean[i]
    elif type(data) is np.ndarray:
        data[np.squeeze(missing_data_flag)] = feature_mean
    else:
        raise ValueError("Sorry. Unrecognized data type")
    return data

def pixmat_between_two_dates(train_date, test_date, path = ""):
    """
    Inputs:
    train_date : Python built-in string
    test_date  : Python built-in string
    path       : Python built-in string
    Output:
    A          : Numpy 174 x 174 array, maps spectra of test_date to train_date
    """
    bbl, _ = get_auxiliary_info(path + "auxiliary_info.mat")
    train_spectra, train_polygons = get_spectra_and_polygon_name(path + train_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    train_coordinates = get_coordinates(path + train_date + "_AVIRIS_speclib_subset_metadata.csv")
    test_spectra, test_polygons = get_spectra_and_polygon_name(path + test_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    test_coordinates = get_coordinates(path + test_date + "_AVIRIS_speclib_subset_metadata.csv")
    A = pixmat(train_spectra, train_polygons, train_coordinates, test_spectra, test_polygons, test_coordinates)
    return A

def randmat_between_two_dates(train_date, test_date, path = ""):
    """
    Inputs:
    train_date : Python built-in string
    test_date  : Python built-in string
    path       : Python built-in string
    Output:
    A          : Numpy 174 x 174 array, maps spectra of test_date to train_date
    """
    bbl, _ = get_auxiliary_info(path + "auxiliary_info.mat")
    train_spectra, train_polygons = get_spectra_and_polygon_name(path + train_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    test_spectra, test_polygons = get_spectra_and_polygon_name(path + test_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    A = randmat(train_spectra, train_polygons, test_spectra, test_polygons)
    return A

def make_dataset_aggregated_polygon(date, path = "../data/"):
    """
    This function aggregates multi-sensor data for same set of polygons of a specific date.
    Input:
    date         :  Python built-in string, e.g., "130411"
    path         :  Python built-in string, path to the raw data file
    
    Output       :  exp1_date_aggregated_dataset.mat saved in path
    
    """
    os.chdir(path)
    ### get bbl info
    bbl, label_name_mapping = get_auxiliary_info("auxiliary_info.mat")
    spectra_raw_csv = date + "_AVIRIS_speclib_subset_spectra.csv"
    ### pixel_level_spectra preprocess
    _, polygon_names = get_spectra_and_polygon_name(spectra_raw_csv, bbl)
    ### get list of unique polygons
    unique_polygons = np.unique(polygon_names)[:, np.newaxis].astype(str)
    ### aggregate spectra
    polygon_spectra = get_spectra(spectra_raw_csv, bbl, unique_polygons)
    ### aggregate thermal
    thermal_raw_csv = date + "_AVIRIS+MASTER_spectral_library_spectra.csv"
    polygon_thermal = get_thermal(thermal_raw_csv, unique_polygons)
    ### aggregate gis features
    gis_raw_csv = "topo_variables_spectral_library_spectra.csv"
    polygon_gis = get_gis_feature(gis_raw_csv, unique_polygons)
    ### split by polygon
    split_indices_csv = date + "_AVIRIS_speclib_subset_trainval.csv"
    split_indices = split_dataset_for_aggregated_polygon(split_indices_csv, polygon_names, unique_polygons, 0)
    split_indices = np.array(split_indices)
    ### assign labels for each labels
    polygon_labels = assign_labels_for_polygons(label_name_mapping, unique_polygons)
    data = [polygon_spectra, polygon_thermal, polygon_gis]
    feature_mean = [None] * 3
    missing_data_flag = [None] * 3
    for i in range(len(data)):
        feature_mean[i], missing_data_flag[i] = get_statistic(data[i])
        data[i] = replace_missing_data(data[i], feature_mean[i], missing_data_flag[i])
    missing_data_flag = np.array(missing_data_flag).squeeze().T
    savemat("exp1_" + date + "_aggregated_dataset.mat", { "unique_polygons":unique_polygons, 
            "polygon_spectra":polygon_spectra, "polygon_thermal":polygon_thermal, 
            "polygon_gis": polygon_gis, "split_indices":split_indices,
            "polygon_labels" : polygon_labels, 
            "missing_data_flag" : missing_data_flag, 
            "spectra_mean": feature_mean[0], "thermal_mean" : feature_mean[1], "gis_mean" : feature_mean[2],
            "spectra_missing_flag" : missing_data_flag[0], "thermal_missing_flag" : missing_data_flag[1], 
            "gis_missing_flag" : missing_data_flag[2]})   

if __name__ == "__main__":
    make_dataset_aggregated_polygon("130411", path = "../data/")