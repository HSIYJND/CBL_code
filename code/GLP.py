#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn import datasets
import sklearn
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.graph import graph_shortest_path


class GLP(object):
    """
    Global Label Propagation Class Implementation based on the papers below with modification.
    [Tenenbaum et al., 2000] Tenenbaum, J. B., De Silva, V., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. science, 290(5500), 2319-2323.
    [Zhu et al., 2002] Zhu, X., & Ghahramani, Z. (2002a). Learning from labeled and unlabeled data with label propagation (Technical Report CMU-CALD-02-107). Carnegie Mellon University.
    [Zhou et al., 2004] Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). Learning with local and global consistency. In Advances in neural information processing systems (pp. 321-328).
    """

    def __init__(self, alpha = 0.99, max_iter = 1000, threshold = 0.001, gamma = 20, 
                       n_neighbors = 7, method = "knn_adj"):
        """
        Initialization Parameters :
        alpha       : absorbing fraction, a positive value in range (0, 1]
                      if alpha is 1, GLP applies hard clamping, otherwise applies soft clamping
        max_iter    : max number of iterations, a positive integer
        threshold   : iteration threshold, a positive value
        n_neighbors : number of neighbors for KNN, a positive interger
        method      : string
                      "knn_adj" : 2002 paper
                      "knn_laplacian" : 2004 paper
                      "knn_shortest" : ISOMAP paper
        """

        assert alpha > 0 and alpha <= 1, "alpha must be greater than 0 and less equal to 1"
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a non-negative integer"
        assert threshold > 0, "threshold must be a positive number"
        assert isinstance(n_neighbors, int) and n_neighbors > 0,  "n_neighbors must be a positive number"

        self.alpha = alpha
        self.max_iter = max_iter
        self.threshold = threshold
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.method = method
    
    def train(self, x, labels):
        """
        Inputs :
        x      : Ntrain x D Numpy array, Ntrain: Number of training samples
                 D : number of features
        labels : Ntrain x 1 Numpy array with labels 0, 1, 2, num_class - 1 
        """
        # fake training, actually no training procedure
        self.train_data = x
        self.train_labels = labels
        self.num_label_sample = len(self.train_data)
        self.label_num = int(len(np.unique(self.train_labels)))
        
        # what if all neighbors of a test point is a unlabeled data ? 
        # keep some statistics information, maybe useful in the future
        self.train_distribution = np.zeros(self.label_num)
        for c in range(self.label_num):
            self.train_distribution[c] = np.where(self.train_labels == c)[0].shape[0] / self.num_label_sample
        
        # fit a KNN model based on the training graph
        self.knn = NearestNeighbors(n_neighbors = self.n_neighbors)
        self.knn.fit(self.train_data)
        
    def test(self, y):
        """
        Input   :
        y       : Ntest x D Numpy array
        Output  :
        predicts: Ntest x 1 Numpy array
        """

        x = self.train_data
        data = np.vstack((x, y))
        self.construct_graph(data)
        num_labeled_sample, num_unlabeled_sample = len(x), len(y)
        num_sample = len(data)
        clamp_label = np.zeros((num_sample, self.label_num))
        
        # for labeled data, label them use a one-hot expression
        # for unlabeled data, the original paper represents them using a zero-vector
        for c in range(self.label_num):
            clamp_label[np.where(self.train_labels == c)[0], c] = 1
        
        #############################################################################################
        # for unlabeled data, find KNN using the train data
        # and initialize its distribution
        for i in range(num_labeled_sample, num_sample):
            _, indices = self.knn.kneighbors(data[i, :].reshape(1, -1))
            neighbor_labels = self.train_labels[indices]
            for c in range(self.label_num):
                clamp_label[i, c] = np.where(neighbor_labels == c)[0].shape[0] / self.n_neighbors
        #############################################################################################

        pre_label_function = clamp_label.copy()
        iter_num, changed = 0, float("inf")
        label_function = np.zeros((num_sample, self.label_num))
        
        while iter_num < self.max_iter and changed > self.threshold:
            # label propagation
            label_function = self.alpha * (self.W @ pre_label_function)
            if self.alpha == 1:
                # hard clamping, reset the labeled data using a one-hot vector
                label_function[: num_labeled_sample, :] = clamp_label[: num_labeled_sample, :]
            else:
                # soft clamping, absorbing the initialized information
                label_function += (1 - self.alpha) * clamp_label
                
            # normalize, each row sums up to 1
            label_function /= np.sum(label_function, axis = 1)[:, np.newaxis]
            
            # check convergence
            changed = np.abs(label_function - pre_label_function).sum()
            pre_label_function = label_function.copy()
            iter_num += 1
            
            if np.any(label_function < 0):
                print("BUG, negative probability")
            

        # final estimation
        label_function /= np.sum(label_function, axis = 1)[:, np.newaxis]
        predicts = np.argmax(label_function[-num_unlabeled_sample:, ], axis = 1)
        
        return predicts.astype(int)

    def construct_graph(self, data):
        """
        Input : 
        data  : (Ntrain + Ntest) x D Numpy array
                first Ntrain rows are training samples and rest are testing samples
        """
        # construct adjency matrix        
        
        if self.method == "knn_adj":
            # global pairwise RBF
            adjency_matrix = rbf_kernel(data, data, gamma = self.gamma)
            self.W = adjency_matrix
            
        elif self.method == "knn_laplacian":
            # normalized laplacian
            adjency_matrix = rbf_kernel(data, data, gamma = self.gamma)
            degree_matrix = np.sqrt(np.sum(adjency_matrix, axis = 1))
            degree_inv_sqrt = np.diagflat(degree_matrix)
            I = np.eye(degree_inv_sqrt.shape[0])
            laplacian_matrix = I - np.linalg.multi_dot([degree_inv_sqrt, adjency_matrix, degree_inv_sqrt])
            self.W = -laplacian_matrix
            for i in range(self.W.shape[0]):
                self.W[i, i] = 0
        elif self.method == "knn_shortest":
            # floyd warshall all-pairs shortest path
            distance_matrix = kneighbors_graph(data, self.n_neighbors, 
                                              mode= "distance", metric="minkowski", 
                                              p=2, metric_params=None, include_self=True).todense()
            distance_matrix += np.eye(len(distance_matrix))
            
            adjacency_matrix = graph_shortest_path(distance_matrix,
                                                method="FW",
                                                directed=False)
            
            self.W = np.exp(-self.gamma * adjacency_matrix**2)
            # set it back, only the connected node in the distance_matrix are connected
            self.W[(distance_matrix == 0)] = 0


def demo(seed = 0, per = 0.5, alpha = 0.99, max_iter = 1000, threshold = 0.001, gamma = 20, 
                       n_neighbors = 7, method = "knn_adj"):
    iris = datasets.load_iris()
    data, labels = iris.data, iris.target
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data)) <= per
    train_data, train_label = data[flag, :], labels[flag]
    test_data, test_target = data[~flag, :], labels[~flag]
    glp = GLP(alpha = alpha, max_iter = max_iter, threshold = threshold, 
                       n_neighbors = n_neighbors, method = method)
    glp.train(train_data, train_label)
    predict = glp.test(test_data)
    accu = np.where(test_target == predict)[0].shape[0] / len(test_target)
    print(accu)
    
    
    return accu

if __name__ == "__main__":
    
    for i in range(10):

        print("adj")
        demo(seed = i, per = 0.5, alpha = 0.99, max_iter = 1000, threshold = 0.001, gamma = 10, 
                           n_neighbors = 7, method = "knn_adj")
        demo(seed = i, per = 0.5, alpha = 1, max_iter = 1000, threshold = 0.001, gamma = 10, 
                           n_neighbors = 7, method = "knn_adj")
        
        print("shortest-path")
        demo(seed = i, per = 0.5, alpha = 0.99, max_iter = 1000, threshold = 0.001, gamma = 10, 
                           n_neighbors = 7, method = "knn_shortest")
    
        demo(seed = i, per = 0.5, alpha = 1, max_iter = 1000, threshold = 0.001, gamma = 10, 
                           n_neighbors = 7, method = "knn_shortest") 

        print("laplacian")
        demo(seed = i, per = 0.5, alpha = 0.99, max_iter = 1000, threshold = 0.001, gamma = 10, 
                          n_neighbors = 7, method = "knn_laplacian")
        demo(seed = i, per = 0.5, alpha = 1, max_iter = 1000, threshold = 0.001, gamma = 10, 
                          n_neighbors = 7, method = "knn_laplacian")
