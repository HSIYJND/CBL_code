from IEEE_paper_nonlinear_mapping_model import NonlinearMappingDataset, NonlinearMappingPerceptron, NonlinearMappingMLP
from IEEE_paper_nonlinear_mapping_utilities import nonlinear_pixmat_helper, nonlinear_randmat_helper
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules.module import Module
import sklearn
from scipy.io import loadmat
import sklearn.neighbors
    
def nonlinear_mapping(alg = "pixmat", train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
                     num_epoch = 3000, activation = "modified_tanh", hidden_unit = 0, linear_init = True):
    """
    Inputs:
    train_date : Python built-in string
    test_date  : Python built-in string
    path       : Python built-in string
    lr         : Python built-in positive float
    num_epoch  : Python built-in integer
    activation : Python built-in string, "tanh", "relu" or "sigmoid"
                 By default we use sigmoid because reflectance is in range [0, 1]
    hidden_unit: Python built-in non-negative integer
                 0 means use single layer perceptron, otherwise use MLP

    Outputs:
    No output but save the trained model
    """
    print("alg:", alg, "train:", train_date, "test:",test_date, "activation:", activation, "linear_init:", linear_init)
    
    # constrcute pairs and linear transformation matrix
    if alg == "pixmat":
        X_train, X_test, A = nonlinear_pixmat_helper(train_date = train_date, test_date = test_date, path = path)
    elif alg == "randmat":
        X_train, X_test, A = nonlinear_randmat_helper(train_date = train_date, test_date = test_date, path = path)
    else:
        print("WRONG alg")
        return
    
    assert isinstance(hidden_unit, int) and hidden_unit >= 0, "hidden_unit is a non-negative number"

    # for custom-define activation function correction
    num_features = X_train.shape[1]
    upper_bound = np.max(X_train)
    init_weight = A if linear_init else None
    init_status = "linear_init" if linear_init else "rand_init"

    if hidden_unit == 0:
        # single layer perceptron
        net = NonlinearMappingPerceptron(num_feature = num_features, activation = activation, 
                                         init_weight = init_weight, upper_bound = upper_bound)
    else:
        # MLP
        net = NonlinearMappingMLP(num_feature = num_features, hidden_unit = hidden_unit, 
                                  activation = activation, init_weight = init_weight, upper_bound = upper_bound)
    
    criterion = nn.MSELoss()  # same criterion as linear mapping
    optimizer = torch.optim.Adam(net.parameters(), lr = lr) 
    
    if torch.cuda.is_available():
        net.cuda() # use GPU
    net.train()

    dataset = NonlinearMappingDataset(X_train, X_test)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size = 512, shuffle = True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): 
            x_train, x_test = data[0].to(device), data[1].to(device)
            x_train, x_test = torch.autograd.Variable(x_train), torch.autograd.Variable(x_test)
            optimizer.zero_grad() # clear previous gradients
            # maps test_date back to train_date!
            x_train_hat = net(x_test)
            
            #########################################################################
            x_train_np = x_train.detach().numpy()
            x_test_np = x_test.detach().numpy()
            x_train_hat_np = x_train_hat.detach().numpy()
            x_train_hat_no_act = np.dot(x_test, A)
            #########################################################################
            
            loss = criterion(x_train_hat, x_train)
            loss.backward() # back propagation
            optimizer.step() # update parameters
            running_loss += loss.item()
            print('# epoch:%d, # batch: %5d loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    model_name = "perceptron" if hidden_unit == 0 else "mlp" + str(hidden_unit)
    
    net.save_state_to_file(path + activation + "_" + init_status + "_" + alg + "_" + model_name + '_from_' + test_date + '_to_' + train_date + '.dat')


def relu_tanh(x, upper_bound = 0.95):
    if upper_bound == 1:
        upper_bound = 0.95
    alpha = (1/(2*upper_bound))*np.log((1 + upper_bound)/(1 - upper_bound))
    x = x.astype(np.float)
    y = np.tanh(alpha*x)
    y[y < 0] = 0
    return y

def nonlinear_mapping_one_batch(alg = "pixmat", train_date = "130411", test_date = "140416", data_path = "../data/", split = 0):
    print("one_batch","alg:", alg, "train:", train_date, "test:",test_date, "split:", split)
    
    # constrcute pairs and linear transformation matrix
    if alg == "pixmat":
        X_train, X_test, A = nonlinear_pixmat_helper(train_date = train_date, test_date = test_date, path = data_path)
    elif alg == "randmat":
        X_train, X_test, A = nonlinear_randmat_helper(train_date = train_date, test_date = test_date, path = data_path)
    else:
        print("WRONG alg")
        return
        
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
    test_spectra_linear = np.dot(test_spectra.copy(), A)
    test_targets = testset["labels"]
    
    test_spectra_nonlinear = relu_tanh(np.dot(test_spectra.copy(), A), np.max(test_spectra))
    test_est_nonlinear = KNN_classifier.predict(test_spectra_nonlinear)
    
    accuracy_nonlinear = round(np.where(test_targets == test_est_nonlinear)[0].shape[0] / len(test_targets) * 100)
    
    print("nonlinear mapping:", accuracy_nonlinear)
    
    # make prediction
    test_est_linear = KNN_classifier.predict(test_spectra_linear)
    accuracy_linear = round(np.where(test_targets == test_est_linear)[0].shape[0] / len(test_targets) * 100)
    print("linear mapping:", accuracy_linear)
    
    
    
def demo():
    for alg in ["pixmat", "randmat"]:
        for train_date, test_date in [("130411", "140416"), ("140416", "130411")]:
            for init_weight in [True, False]:
                nonlinear_mapping(alg = alg, train_date = train_date, test_date = test_date, path = "../data/", lr = 0.001, 
                     num_epoch = 300, activation = "modified_relu_tanh", hidden_unit = 0, linear_init = init_weight)

def demo_one_batch():
    for i in range(5):
        for alg in ["pixmat", "randmat"]:
            for train_date, test_date in [("130411", "140416"), ("140416", "130411")]:
                nonlinear_mapping_one_batch(alg = alg, train_date = train_date, 
                                            test_date = test_date, data_path = "../data/", split = i)

#if __name__ == "__main__":
#    demo()
#    demo_one_batch()
    # nonlinear_mapping_one_batch
    
#    
#    ########## FORWARD #########################################################################
##    nonlinear_pixmat(alg = "pixmat", train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
##                     num_epoch = 300, activation = "modified_relu_tanh", hidden_unit = 0, linear_init = True)
##    
##    nonlinear_pixmat(alg = "pixmat", train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
##                     num_epoch = 300, activation = "modified_tanh", hidden_unit = 0, linear_init = True)
##    
#    ########### BACKWARD ###########################################################################
#    print("Backward mapping without linear transfromation initialization")
#    
#    ############# randmat #######################################################################################
#    nonlinear_mapping(alg = "pixmat", train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
#                     num_epoch = 300, activation = "modified_relu_tanh", hidden_unit = 0, linear_init = True)
#    
#    nonlinear_mapping(alg = "randmat", train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
#                     num_epoch = 300, activation = "modified_relu_tanh", hidden_unit = 0, linear_init = True)
#    
#    ########### BACKWARD ###########################################################################
#    nonlinear_mapping(alg = "pixmat", train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
#                     num_epoch = 300, activation = "modified_relu_tanh", hidden_unit = 0, linear_init = True)
#    
#    nonlinear_mapping(alg = "randmat", train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
#                     num_epoch = 300, activation = "modified_relu_tanh", hidden_unit = 0, linear_init = True)
#    
#    
##    nonlinear_pixmat(train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
##                     num_epoch = 100, activation = "modified_tanh", hidden_unit = 0)
##    nonlinear_randmat(train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
##                     num_epoch = 100, activation = "modified_tanh", hidden_unit = 0)
#    ########## BACKWARD ########################################################################
#    
##    nonlinear_pixmat(train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
##                     num_epoch = 100, activation = "tanh", hidden_unit = 0)
##    nonlinear_randmat(train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
##                     num_epoch = 100, activation = "tanh", hidden_unit = 0)
##    
##    nonlinear_pixmat(train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
##                     num_epoch = 100, activation = "sigmoid", hidden_unit = 0)
##    nonlinear_randmat(train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
##                     num_epoch = 100, activation = "sigmoid", hidden_unit = 0)
#    
#    
##    for i in range(3, 180, 3):
##        ########## FORWARD #########################################################################
##        nonlinear_pixmat(train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
##                         num_epoch = 300, activation = "sigmoid", hidden_unit = i)
##        nonlinear_randmat(train_date = "130411", test_date = "140416", path = "../data/", lr = 0.001, 
##                         num_epoch = 300, activation = "sigmoid", hidden_unit = i)
##        ########## BACKWARD ########################################################################
##        nonlinear_pixmat(train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
##                         num_epoch = 300, activation = "sigmoid", hidden_unit = i)
##        nonlinear_randmat(train_date = "140416", test_date = "130411", path = "../data/", lr = 0.001, 
##                         num_epoch = 300, activation = "sigmoid", hidden_unit = i)