#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:07:45 2019

@author: yuanhang
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat

def get_auxiliary_info(file_name):
    """
    Input:
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

def nonlinear_pixmat_helper(train_date = "130411", test_date = "140416", path = "../data/"):
    """
    Inputs:
    train_date : Python built-in string
    test_date  : Python built-in string
    path       : Python built-in string

    Outputs:
    X_train    : Npixels x 174 spectra of train date
    X_test     : Npixels x 174 spectra of test date
    """

    bbl, _ = get_auxiliary_info(path + "auxiliary_info.mat")
    train_spectra, train_polygons = get_spectra_and_polygon_name(path + train_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    train_coordinates = get_coordinates(path + train_date + "_AVIRIS_speclib_subset_metadata.csv")
    test_spectra, test_polygons = get_spectra_and_polygon_name(path + test_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    test_coordinates = get_coordinates(path + test_date + "_AVIRIS_speclib_subset_metadata.csv")

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
    A = np.linalg.lstsq(X_test, X_train, rcond = None)[0] 
    return X_train, X_test, A

def nonlinear_randmat_helper(train_date = "130411", test_date = "140416", path = "../data/"):
    """
    Inputs:
    train_date : Python built-in string
    test_date  : Python built-in string
    path       : Python built-in string

    Outputs:
    X_train    : Npixels x 174 spectra of train date
    X_test     : Npixels x 174 spectra of test date
    """

    bbl, _ = get_auxiliary_info(path + "auxiliary_info.mat")
    train_spectra, train_polygons = get_spectra_and_polygon_name(path + train_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
    test_spectra, test_polygons = get_spectra_and_polygon_name(path + test_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)

    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []
     # Step 2 : iterate all the common polygons and construct pairs by randomly selection
     # using uniform distribution
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
    A = np.linalg.lstsq(X_test, X_train, rcond = None)[0] 
    return X_train, X_test, A

nonlinear_pixmat_helper()