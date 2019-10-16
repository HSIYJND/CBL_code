#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:58:55 2019

@author: yuanhang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
from torch.nn.modules.module import Module

#%% CUSTOM ACTIVATION FUNCTION ##
class ModifiedTanh(Module):
    def __init__(self, upper_bound = 0.9):
        super(ModifiedTanh, self).__init__()
        if upper_bound == 1:
            upper_bound = 0.95
        self.upper_bound = upper_bound
        self.alpha = (1/(2*upper_bound))*np.log((1 + upper_bound)/(1 - upper_bound))
#        print("check point:", "input:", upper_bound, "output:", self(torch.FloatTensor([upper_bound])))
    
    def forward(self, input):
        return torch.tanh(self.alpha * input)

class ModifiedReLuTanh(Module):
    def __init__(self, upper_bound = 0.9):
        super(ModifiedReLuTanh, self).__init__()
        if upper_bound == 1:
            upper_bound = 0.95
        self.upper_bound = upper_bound
        self.alpha = (1/(2*upper_bound))*np.log((1 + upper_bound)/(1 - upper_bound))
#        print("check point:", "input:", upper_bound, "output:", self(torch.FloatTensor([upper_bound])))
        
    def forward(self, input):
        return torch.nn.functional.relu((torch.tanh(self.alpha * input)))

class ModifiedSigmoid(Module):
    def __init__(self):
        super(ModifiedSigmoid, self).__init__()
        self.lower_bound = 0.001
        self.alpha = -1 * np.log(1/self.lower_bound - 1)
    
    def forward(self, input):
        torch.sigmoid(input + self.alpha)

#%% MODELS
class NonlinearMappingPerceptron(nn.Module):
    
    def __init__(self, num_feature = 174, activation = "sigmoid", 
                       init_weight = None, upper_bound = 0.9):
        # reflectance are in range[0, 1] so I use sigmoid by default

        super(NonlinearMappingPerceptron, self).__init__()
        self.fc = nn.Linear(num_feature, num_feature, bias = False)
        if init_weight is not None:
            # pytorch linear function : output = input.matmul(weight.t())
            print("initialize using linear transformation matrix")
            self.fc.weight.data = torch.nn.Parameter(torch.FloatTensor(init_weight)).t()
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace = False) # I want to have a look at the output
        elif activation == "modified_tanh":
            self.activation = ModifiedTanh(upper_bound = upper_bound)
        elif activation == "modified_relu_tanh":
            self.activation = ModifiedReLuTanh(upper_bound = upper_bound)
        else:
            raise ValueError("Unsupported Activation Function! Only support tanh, sigmoid and relu")

    def forward(self, x):
        y = self.fc(x)
        y = self.activation(y) # add non-linearity
        return y
    
    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))
        
    def rectified_tanh(self, input):
        pass

class NonlinearMappingMLP(nn.Module):
    
    def __init__(self, num_feature = 174, hidden_unit = 87, activation = "sigmoid", 
                 init_weight = None, upper_bound = 0.9):
        # reflectance are in range[0, 1] so I use sigmoid by default
        
        super(NonlinearMappingMLP, self).__init__()
        self.fc1 = nn.Linear(num_feature, hidden_unit, bias = False)
        if init_weight is not None:
            print("initialize using linear transformation matrix")
            self.fc1.weight.data = torch.nn.Parameter(torch.FloatTensor(init_weight)).t()
        self.fc2 = nn.Linear(hidden_unit, num_feature, bias = False)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace = False) # I want to have a look at the output
        elif activation == "modified_tanh":
            self.activation = ModifiedTanh(upper_bound = upper_bound)
        elif activation == "modified_relu_tanh":
            self.activation = ModifiedReLuTanh(upper_bound = upper_bound)
        else:
            raise ValueError("Unsupported Activation Function! Only support tanh, sigmoid and relu")

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.activation(y)
        return y
    
    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))

#%% DATASET
class NonlinearMappingDataset(tdata.Dataset):
    
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        # x_train, x_test
        # try to map x_test to x_train
        # that is, x_train_hat = f(x_test) ~ x_train
        return torch.from_numpy(self.train_data[index,:]).float(), \
               torch.from_numpy(self.test_data[index,:]).float() 