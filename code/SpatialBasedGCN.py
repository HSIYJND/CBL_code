import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F
import torch.utils.data as tdata

class SpatialBasedGraphConvLayer(Module):

    def __init__(self, in_features, out_features, bias = True):
        super(SpatialBasedGraphConvLayer, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self._bias = Parameter(torch.FloatTensor(out_features))
        else:
            self._bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self._weight.size(1))
        self._weight.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self._weight)
        output = torch.mm(adj, output)
        if self._bias is not None:
            return output + self._bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._in_features) + ' -> ' \
               + str(self._out_features) + ')'

class SpatialBasedGraphConvNet(Module):
    
    def __init__(self, num_feature = 10, num_hidden = 10, num_class = 27, 
                 dropout = 0.1):
        
        super(SpatialBasedGraphConvNet, self).__init__()
        self._conv1 = SpatialBasedGraphConvLayer(num_feature, num_hidden)
        self._conv2 = SpatialBasedGraphConvLayer(num_hidden, num_class)
        self._dropout = dropout

    def forward(self, x, adj):
        y = F.relu(self._conv1(x, adj))
        y = F.dropout(y, p = self._dropout, training = self.training)
        y = self._conv2(y, adj)
        return F.log_softmax(y, dim = 1)
    
    def semi_supervised_training_and_testing(self, data, label, kernel = "knn", n_neighbors = 30,
                                             gamma = 20, num_epoch = 10000, lr = 0.001):
        """
        Given the training and testing data together.
        Train the GCN using labeled data and test on the unlabeled data.
        
        Inputs:
        data        : (Nsample x Nfeatures) Numpy array   
        label       : (Nsample,) Numpy array, -1(unlabeled), 0,1,2,...,Nclass - 1
        kernel      : Python String, "knn" / "rbf"
        n_neighbors : int, k for KNN
        gamma       : float, gamma for rbf kernel
        num_epoch   : int
        lr          : learning rate
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        
        import sklearn.neighbors
        
        if kernel == "knn":
            adj = sklearn.neighbors.kneighbors_graph(data, n_neighbors, mode = "connectivity", 
                                                     metric = "minkowski", p = 2, metric_params = None, 
                                                     include_self = True).toarray()
            self.save_trainset(data, adj)
            
        elif kernel == "rbf":
            adj = sklearn.neighbors.kneighbors_graph(data, n_neighbors = n_neighbors, mode = "distance", 
                                                     metric = "minkowski", p = 2, metric_params = None, 
                                                     include_self = True).toarray()
            adj[adj == 0] = np.inf
            adj *= -gamma
            np.exp(adj, adj)
            adj += np.eye(len(data))
            
            self.save_trainset(data, adj)
            
            # Laplacian Matrix
            D = np.diag(np.sum(adj, axis = 1))
            L = D - adj
            adj = np.sqrt(D) @ L @ np.sqrt(D)
        
        # check type
        if not isinstance(data, torch.FloatTensor):
            data = torch.FloatTensor(data)
        
        if not isinstance(label, torch.LongTensor):
            train_idx = torch.LongTensor(np.where(label != -1)[0]) 
            test_idx = torch.LongTensor(np.where(label == -1)[0])
            label = torch.LongTensor(label)
        
        adj = adj / np.sum(adj, axis = 1)[:, np.newaxis]
        
        #############################################################
#        print("Laplacian")
#        D = np.eye(adj.shape[0])
#        L = D - adj
#        L_norm = np.sqrt(D) @ L @ np.sqrt(D)
#        adj = L_norm.copy()        
        #############################################################
        
        adj = torch.FloatTensor(adj)
        
         # use GPU to acclerate training/testing
        if torch.cuda.is_available():
            data = data.cuda()
            adj = adj.cuda()
            label = label.cuda()
            self.cuda()
        
        self.train()
        best_loss = float("inf")
        
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            output = self.forward(data, adj)
            loss = F.nll_loss(output[train_idx], label[train_idx])
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print("Epoch:",epoch,",Loss:",loss.cpu().detach().numpy())
            if loss.cpu().data < best_loss:
                best_loss = loss.cpu().data
                self.save_state_to_file("SpatialBasedGCN.dat")
        
        self.eval()
        output = self.forward(data, adj)
        predicts = output.max(1)[1][test_idx]
        return predicts
    
    def supervised_training(self, data, label, kernel = "knn", n_neighbors = 30,
                                  gamma = 20, num_epoch = 10000, lr = 0.001):
        """
        Given the training data only, train the GCN.
        Inputs:
        data        : (Nsample x Nfeatures) Numpy array   
        label       : (Nsample,) Numpy array, -1(unlabeled), 0,1,2,...,Nclass - 1
        kernel      : Python String, "knn" / "rbf"
        n_neighbors : int, k for KNN
        gamma       : float, gamma for rbf kernel
        num_epoch   : int
        lr          : learning rate
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        
        import sklearn.neighbors
        
        if kernel == "knn":
            adj = sklearn.neighbors.kneighbors_graph(data, n_neighbors = n_neighbors, mode = "connectivity", 
                                                     metric = "minkowski", p = 2, metric_params = None, 
                                                     include_self = True).toarray()
            self.save_trainset(data, adj)
            
        elif kernel == "rbf":
            adj = sklearn.neighbors.kneighbors_graph(data, n_neighbors = n_neighbors, mode = "distance", 
                                                     metric = "minkowski", p = 2, metric_params = None, 
                                                     include_self = True).toarray()
            adj[adj == 0] = np.inf
            adj *= -gamma
            np.exp(adj, adj)
            adj += np.eye(len(data))
            
            self.save_trainset(data, adj)
            
            # Laplacian Matrix
            D = np.diag(np.sum(adj, axis = 1))
            L = D - adj
            adj = np.sqrt(D) @ L @ np.sqrt(D)
                
        # check type
        if not isinstance(data, torch.FloatTensor):
            data = torch.FloatTensor(data)

        if not isinstance(label, torch.LongTensor):
            label = torch.LongTensor(label)
        
        adj = adj / np.sum(adj, axis = 1)[:, np.newaxis]
        adj = torch.FloatTensor(adj)
        
         # use GPU to acclerate training/testing
        if torch.cuda.is_available():
            data = data.cuda()
            adj = adj.cuda()
            label = label.cuda()
            self.cuda()
        
        self.train()

        best_loss = float("inf")
        
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            output = self.forward(data, adj)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print("Epoch:",epoch,",Loss:",loss.cpu().detach().numpy())
            if loss.cpu().data < best_loss:
                best_loss = loss.cpu().data
                self.save_state_to_file("SpatialBasedGCN.dat")
    
    def save_trainset(self, data, adj):
        self.train_data = data.copy()
        self.train_adj = adj.copy()

    def test_individual_sample(self, test_data, method = "replacement", 
                               kernel = "knn", n_neighbors = 30, gamma = 20):
        """
        During the training process, testing data is unseen.
        Encode the new testing sample back to the training manifold, 
        and then use the pretrained GCN to do classification.
        
        Input:
        test_data   : (Ntest x Nfeatures) Numpy array   
        method      : "replacement" --- only change weight but do not add/delete edges
                      "embedding"--- change weight, may add/delete edges
        kernel      : Python String, "knn" / "rbf"
        n_neighbors : int, k for KNN
        gamma       : float, gamma for rbf kernel
        """
        
        predicts = torch.zeros(len(test_data))
        self.load_state_from_file("SpatialBasedGCN.dat")
        
        import sklearn
        
        for i in range(len(test_data)):
            self.eval()
            
            # pairwise distance
            dist = sklearn.metrics.pairwise_distances(self.train_data, test_data[i, :].reshape(1, -1)).squeeze()
            
            # sort the pairwise distance in ascending order
            order = (np.argsort(dist))[:n_neighbors]
            
            # find best matching unit
            bmu_idx = order[0]
            new_adj = self.train_adj.copy()
            
            if method == "replacement":
            
                if kernel == "knn":
                    # do nothing, we don't need to modify the manifold because all edges are unweighted, i.e., 1
                    # simply replace feature matrix
                    pass

                elif kernel == "rbf":
                    # modify weights only, do not add/delete edges
                    dist *= -gamma
                    np.exp(dist, dist)
                    previous_edge_idx = np.where(new_adj[bmu_idx, :] != 0)[0]
                    for edge_idx in previous_edge_idx:
                        if edge_idx == bmu_idx:
                            new_adj[bmu_idx, edge_idx] = 1
                        else:
                            new_adj[bmu_idx, edge_idx] = dist[edge_idx]
                            new_adj[edge_idx, bmu_idx] = new_adj[bmu_idx, edge_idx]

            elif method == "embedding":

                if kernel == "knn":
                    # clear the row
#                    prev_adj = new_adj.copy()
                    new_adj[bmu_idx, :] = 0
                    # modify the manifold by changing positions of 1 in bmu row
                    # actually still has a one on the diagnonal
                    for i in range(len(order)):
                        new_adj[bmu_idx, order[i]] = 1
                        new_adj[order[i], bmu_idx] = new_adj[bmu_idx, order[i]] 
                    
#                    print(np.abs(new_adj - prev_adj).sum())
                
                elif kernel == "rbf":
                    # clear the row
                    new_adj[bmu_idx, :] = 0
                    # modify weights only and add/delete edges
                    dist *= -gamma
                    np.exp(dist, dist)
                    # doesn't have a one on the diagnonal
                    for i in range(len(order)):
                        new_adj[bmu_idx, order[i]] = dist[order[i]]
                        new_adj[order[i], bmu_idx] = new_adj[bmu_idx, order[i]] 
                    
                    # Laplacian Matrix
                    D = np.diag(np.sum(new_adj, axis = 1))
                    L = D - new_adj
                    new_adj = np.sqrt(D) @ L @ np.sqrt(D)
            
            # replace feature matrix
            new_data = self.train_data.copy()
            new_data[bmu_idx] = test_data[i, :]
            
            new_data = torch.FloatTensor(new_data)
            
            new_adj /= np.sum(new_adj, axis = 1)[:, np.newaxis]
            new_adj = torch.FloatTensor(new_adj)
            
            if torch.cuda.is_available():
                new_data = new_data.cuda()
                new_adj = new_adj.cuda()
            
            output = self.forward(new_data, new_adj)
            predicts[i] = output.max(1)[1][bmu_idx]
            
        return predicts
            
    def test_new_manifold(self, test_data, kernel = "knn", 
                          n_neighbors = 30, gamma = 20):
        """
        Given a new different manifold.
        Without encoding the testing samples, simply use the pretrained filters for testing.
        
        Input:
        test_data  :   (Ntest x Nfeatures) Numpy array   
        kernel     :   "knn" / "rbf"
        """
        if kernel == "knn":
            import sklearn
            test_adj = sklearn.neighbors.kneighbors_graph(test_data, n_neighbors = n_neighbors, mode = "connectivity", 
                                                     metric = "minkowski", p = 2, metric_params = None, 
                                                     include_self = True).toarray()
            
        elif kernel == "rbf":
            import sklearn
            test_adj = sklearn.neighbors.kneighbors_graph(test_data, n_neighbors = n_neighbors, mode = "distance", 
                                                     metric = "minkowski", p = 2, metric_params = None, 
                                                     include_self = True).toarray()
            test_adj[test_adj == 0] = np.inf
            test_adj *= -gamma
            np.exp(test_adj, test_adj)
            test_adj += np.eye(len(test_data))
        
        test_data = torch.FloatTensor(test_data)
        test_adj /= np.sum(test_adj, axis = 1)
        test_adj = torch.FloatTensor(test_adj)
        
        if torch.cuda.is_available():
            test_data = test_data.cuda()
            test_adj = test_adj.cuda()
        
        self.load_state_from_file("SpatialBasedGCN.dat")
        self.eval()
        output = self.forward(test_data, test_adj)
        return output.max(1)[1]

    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))

def demo_iris_semi_supervised_training(seed = 0, unlabeled_rate = 0.2):
    from sklearn import datasets
    import sklearn.neighbors
    iris = datasets.load_iris()
    data, label = iris.data, iris.target
    net = SpatialBasedGraphConvNet(4, 3, 3)
    partial_label = label.copy()
    rng = np.random.RandomState(seed)
    idx = np.where(rng.rand(len(data)) < unlabeled_rate)[0]
    partial_label[idx] = -1
    predicts = net.semi_supervised_training_and_testing(data, partial_label, kernel = "knn", n_neighbors = 6,gamma = 10,
                                                        num_epoch = 10000, lr = 0.01).cpu().numpy()
    targets = label[idx]
    print("iris_semi_supervised_training accuracy:", sklearn.metrics.accuracy_score(targets, predicts))

def demo_iris_supervised_training_testing(seed = 0, unlabeled_rate = 0.6):
    from sklearn import datasets
    import sklearn.neighbors
    iris = datasets.load_iris()
    data, label = iris.data, iris.target
    
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data)) < unlabeled_rate
    
    train_data, train_label = data[~flag, :], label[~flag]
    
    net = SpatialBasedGraphConvNet(4, 3, 3)
    net.supervised_training(train_data, train_label, kernel = "knn", n_neighbors = 6,
                            num_epoch = 10000, lr = 0.001)
    
    test_data, test_label = data[flag, :], label[flag]
    
    predicts = net.test_individual_sample(test_data, method = "embedding", 
                               kernel = "knn", n_neighbors = 6).cpu().numpy()
    
    print("embedding accuracy:", sklearn.metrics.accuracy_score(test_label, predicts))


    predicts = net.test_individual_sample(test_data, method = "replacement", 
                               kernel = "knn", n_neighbors = 6).cpu().numpy()
    
    print("replacement accuracy:", sklearn.metrics.accuracy_score(test_label, predicts))

def demo_susan_dataset_aggregated_semi_supervised_training(seed = 0, test_rate = 0.8):
    from scipy.io import loadmat
    import sklearn.neighbors
    from sklearn.metrics.pairwise import rbf_kernel
    x = loadmat("../data/" + "exp1_130411_aggregated_dataset", squeeze_me = True)
    data, label = x["polygon_spectra"], x["polygon_labels"]
    
    
    partial_label = label.copy()
    rng = np.random.RandomState(seed)
    idx = np.where(rng.rand(len(data)) < test_rate)[0]
    partial_label[idx] = -1
    
    adj = rbf_kernel(data, data, gamma = 20)
    
    D = np.diag(np.sum(adj, axis = 1))
    L = np.sqrt(D) @ adj @ np.sqrt(D)
    adj = L.copy()
    
    net = SpatialBasedGraphConvNet(26, 21, 27, 0.1)
    
    predicts = net.semi_supervised_training_and_testing(data, adj, partial_label, 
                                                        num_epoch = 10000, lr = 0.001).cpu().numpy()
    targets = label[idx]

    print("accuracy:", sklearn.metrics.accuracy_score(targets, predicts))

def demo_susan_dataset_supervised_training_testing(seed = 0, unlabeled_rate = 0.2):
    from scipy.io import loadmat
    import sklearn
    
    x = loadmat("../data/" + "exp1_130411_aggregated_dataset", squeeze_me = True)
    data, label = x["polygon_spectra"], x["polygon_labels"]
    data[data < 0] = 0
    data[data > 1] = 1
    
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data)) < unlabeled_rate
    
    train_data, train_label = data[~flag, :], label[~flag]
    test_data, targets = data[flag, :], label[flag]
    rng = np.random.RandomState(seed)
    test_idx = np.where(rng.rand(len(data)) < unlabeled_rate)[0]
    
    partial_label = label.copy()
    partial_label[test_idx] = -1
    
    net = SpatialBasedGraphConvNet(174, 88, 27)
    net.supervised_training(train_data, train_label, kernel = "knn", n_neighbors = 15,
                            num_epoch = 100000, lr = 0.001)
    
    predicts = net.test_individual_sample(test_data, method = "replacement", 
                                          kernel = "knn", n_neighbors = 15).cpu().numpy()
    print("accuracy:", sklearn.metrics.accuracy_score(targets, predicts))

def parameter_tuning(seed = 0, nhid = 10, unlabeled_rate = 0.2):
    from scipy.io import loadmat
    import sklearn
    
    x = loadmat("../data/" + "exp1_130411_aggregated_dataset", squeeze_me = True)
    data, label = x["polygon_spectra"], x["polygon_labels"]
    data[data < 0] = 0
    data[data > 1] = 1
    
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data)) < unlabeled_rate
    
    train_data, train_label = data[~flag, :], label[~flag]
    test_data, targets = data[flag, :], label[flag]
    
    print("with cda")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_data, train_label)  
    train_data = clf.transform(train_data)
    test_data = clf.transform(test_data)
    
    net = SpatialBasedGraphConvNet(train_data.shape[1], nhid, 27)
    net.supervised_training(train_data, train_label, kernel = "knn", n_neighbors = 15,
                            num_epoch = 10000, lr = 0.001)
    
    predicts = net.test_individual_sample(test_data, method = "replacement", 
                                          kernel = "knn", n_neighbors = 15).cpu().numpy()
    print("nhid:", nhid, "accuracy:", sklearn.metrics.accuracy_score(targets, predicts))

def parameter_tuning_semi_supervised(seed = 0, nhid = 21, unlabeled_rate = 0.2):
    from scipy.io import loadmat
    import sklearn
    
    x = loadmat("../data/" + "exp1_130411_aggregated_dataset", squeeze_me = True)
    data, label = x["polygon_spectra"], x["polygon_labels"]
    data[data < 0] = 0
    data[data > 1] = 1
    
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data))
    train_idx = np.where(flag >= unlabeled_rate)[0]
    test_idx = np.where(flag < unlabeled_rate)[0]
    
    partial_label = label.copy()
    partial_label[test_idx] = -1

    print("semi_supervised with cda")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(data[train_idx,:], label[train_idx])  
    data = clf.transform(data)

    net = SpatialBasedGraphConvNet(data.shape[1], nhid, 27)

    predicts = net.semi_supervised_training_and_testing(data, partial_label, kernel = "knn", n_neighbors = 6,gamma = 10,
                                                        num_epoch = 10000, lr = 0.01).cpu().numpy()
    targets = label[test_idx]

    print("semi_supervised_training accuracy:", sklearn.metrics.accuracy_score(targets, predicts))


def pixel_level_semi_supervised_learning_parameter_tuning(seed = 0, nhid = 10, unlabeled_rate = 0.5):
    from scipy.io import loadmat
    
    x = loadmat("../data/" + "SusanSpectraProcessed130411.mat", squeeze_me = True)
    
    data, label = x["spectra"], x["labels"]
    label -= 1
    data[data < 0] = 0
    data[data > 1] = 1
    data = data[:, x["bbl"] == 1]
    data = data[:, 2:]
    
    rng = np.random.RandomState(seed)
    test_idx = np.where(rng.rand(len(data)) < unlabeled_rate)[0]
    
    partial_label = label.copy().astype(np.int)
    partial_label[test_idx] = -1

    
    net = SpatialBasedGraphConvNet(174, nhid, 30)

    predicts = net.semi_supervised_training_and_testing(data, partial_label, kernel = "knn", n_neighbors = 15,gamma = 10,
                                                        num_epoch = 10000, lr = 0.01).cpu().numpy()
    
    targets = label[test_idx]
    
    import sklearn
    print("nhid:", nhid, "accuracy:", sklearn.metrics.accuracy_score(targets, predicts))

def pixel_level_supervised_learning_parameter_tuning(seed = 0, nhid = 10, unlabeled_rate = 0.8):
    from scipy.io import loadmat
    import sklearn
    
    x = loadmat("../data/" + "SusanSpectraProcessed130411.mat", squeeze_me = True)
    
    data, label = x["spectra"], x["labels"]
    label -= 1
    data[data < 0] = 0
    data[data > 1] = 1
    data = data[:, x["bbl"] == 1]
    data = data[:, 2:]
    
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data)) < unlabeled_rate
    
    train_data, train_label = data[~flag, :], label[~flag]
    test_data, targets = data[flag, :], label[flag]
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_data, train_label)  
    train_data = clf.transform(train_data)
    test_data = clf.transform(test_data)
    
    net = SpatialBasedGraphConvNet(train_data.shape[1], nhid, 30)
    net.supervised_training(train_data, train_label, kernel = "knn", n_neighbors = 30,
                            num_epoch = 10000, lr = 0.001)
    
    print("predicting...")
    predicts = net.test_individual_sample(test_data, method = "replacement", 
                                          kernel = "knn", n_neighbors = 30).cpu().numpy()
    
    print("nhid:", nhid, "accuracy:", sklearn.metrics.accuracy_score(targets, predicts))

def demo_cora_dataset_supervised_training_testing():
    from scipy.io import loadmat
    import sklearn
    
    x = loadmat("../data/" + "cora.mat", squeeze_me = True)
    data, labels = x["features"], x["labels"]
    train_idx = x["idx_test"]
    test_idx = x["idx_train"]
    
    net = SpatialBasedGraphConvNet(data.shape[1], 16, labels.max() + 1, 0.2)
    net.supervised_training(data[train_idx,:], labels[train_idx], kernel = "knn", n_neighbors = 30,
                            num_epoch = 1000, lr = 0.01)
    
    predicts = net.test_individual_sample(data[test_idx, :], method = "replacement", 
                                          kernel = "knn", n_neighbors = 30).cpu().numpy()
    
    print("cora accuracy:", sklearn.metrics.accuracy_score(labels[test_idx], predicts))

def demo_cora_dataset_semi_supervised_training_testing():
    from scipy.io import loadmat
    import sklearn
    
    x = loadmat("../data/" + "cora.mat", squeeze_me = True)
    data, labels = x["features"], x["labels"]

    test_idx = x["idx_train"]
    
    partial_labels = labels.copy()
    partial_labels[test_idx] = -1
    
    net = SpatialBasedGraphConvNet(data.shape[1], 16, labels.max() + 1, 0.2)
    predicts = net.semi_supervised_training_and_testing(data, partial_labels, kernel = "knn", n_neighbors = 30,
                            num_epoch = 1000, lr = 0.01).cpu().numpy()
    
    
    print("cora accuracy:", sklearn.metrics.accuracy_score(labels[test_idx], predicts))

if __name__ == "__main__":
#    demo_iris_semi_supervised_training()
#    demo_iris_supervised_training_testing()
#    demo_cora_dataset_supervised_training_testing()
#    demo_cora_dataset_semi_supervised_training_testing()
#    demo_cora_dataset_supervised_training_testing()
    
#    demo_susan_dataset()
#    demo_susan_dataset_supervised_training_testing()
    
#    for i in range(10, 30, 3):
#        pixel_level_supervised_learning_parameter_tuning(seed = 0, nhid = 10, unlabeled_rate = 0.5)
    
    for nhid in range(15, 27, 3):
        for i in range(10):
            print("seed:",i, "nhid:",nhid)
            parameter_tuning_semi_supervised(seed = i, nhid = nhid, unlabeled_rate = 0.5)
    
    #for nhid in range(15, 27, 3):
    #    for i in range(10):
    #        print("seed:",i, "nhid:",nhid)
    #        parameter_tuning(seed = i, nhid = nhid, unlabeled_rate = 0.5)
    
    

#    for nhid in range(10, 170, 10):
#        pixel_level_semi_supervised_learning_parameter_tuning()