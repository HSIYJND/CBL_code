#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from cvxopt import solvers, matrix
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances

class LLP(object):
    """
    Local Label Propagation Class Implementation based on the papers below with modification.
    [Roweis et al., 2000] Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. science, 290(5500), 2323-2326.
    [Wang et al., 2007] Wang, F., & Zhang, C. (2007). Label propagation through linear neighborhoods. 
    IEEE Transactions on Knowledge and Data Engineering, 20(1), 55-67.
    """

    def __init__(self, alpha = 0.99, max_iter = 50, threshold = 0.001, 
                 n_neighbors = 7, allow_negative = False):
        """
        Initialization Parameters :
        alpha       : absorbing fraction, a positive value in range (0, 1]
                      if alpha is 1, LLP applies hard clamping, otherwise applies soft clamping
        max_iter    : max number of iterations, a positive integer
        threshold   : iteration threshold, a positive value
        n_neighbors : number of neighbors for KNN, a positive interger
        """

        assert alpha > 0 and alpha <= 1, "alpha must be greater than 0 and less equal to 1"
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a non-negative integer"
        assert threshold > 0, "threshold must be a positive number"
        assert isinstance(n_neighbors, int) and n_neighbors > 0,  "n_neighbors must be a positive number"

        self.alpha = alpha
        self.max_iter = max_iter
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.allow_negative = allow_negative
    
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
        
        # what if all neighbors of a test point is a unlabeled data
        # keep some statistics information
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
        # for labeled data
        for c in range(self.label_num):
            clamp_label[np.where(self.train_labels == c)[0], c] = 1
        
        ########################################################################################
        # for unlabeled data, find KNN using the train data
        # and initialize its distribution
        # original paper initialize with 0 
        for i in range(num_labeled_sample, num_sample):
            _, indices = self.knn.kneighbors(data[i, :].reshape(1, -1))
            neighbor_labels = self.train_labels[indices]
            for c in range(self.label_num):
                clamp_label[i, c] = np.where(neighbor_labels == c)[0].shape[0] / self.label_num
        ########################################################################################

        pre_label_function = clamp_label.copy()
        iter_num, changed = 0, float("inf")
        label_function = np.zeros((num_sample, self.label_num))
                
        while iter_num < self.max_iter and changed > self.threshold:
            
            label_function = self.alpha * (self.W @ pre_label_function)
            
            if self.alpha == 1:
                # hard clamping, reset the labeled data using a one-hot vector
                label_function[: num_labeled_sample, :] = clamp_label[: num_labeled_sample, :]
            else:
                # soft clamping, absorbing the initialized information
                label_function += (1 - self.alpha) * clamp_label
            
            # check convergence
            changed = np.abs(label_function - pre_label_function).sum()
            pre_label_function = label_function.copy()
            iter_num += 1
        
        predicts = np.argmax(label_function[-num_unlabeled_sample:, ], axis = 1)
        return predicts.astype(int)

    def construct_graph(self, data):
        """
        Input : 
        data  : (Ntrain + Ntest) x D Numpy array
                first Ntrain rows are training samples and rest are testing samples
        """
        # construct adjency matrix
        adjency_matrix = kneighbors_graph(data, self.n_neighbors, mode='connectivity', 
                                          include_self = False).todense()
        self.W = np.zeros((len(data), len(data)))
        
        # do not print QP iteration information
        solvers.options['show_progress'] = False
        for i in range(len(data)):
            # find out K nearest neighbors of data i
            neighbor_indices = np.where(adjency_matrix[i, :].T != 0)[0]
            neighbors = data[neighbor_indices, :]
            x_i = data[i, :] 
            # initialize local gram, a K by K matrix
            local_gram = np.zeros((self.n_neighbors, self.n_neighbors))
            for j in range(self.n_neighbors):
                x_j = neighbors[j, :]
                for k in range(j, self.n_neighbors):
                    x_k = neighbors[k, :]
                    local_gram[j, k] = np.inner(x_i - x_j, x_i - x_k)
                    # local gram matrix is symmetric
                    local_gram[k, j] = local_gram[j, k]
                                
            # now we have our local Gram matrix and two constraints
            if not self.allow_negative:
                P = local_gram.copy()
                q = np.zeros((self.n_neighbors, 1))
                # non-negative weights sums up to 1
                G = -1 * np.eye(self.n_neighbors)
                h = np.zeros((self.n_neighbors, 1))
                A = np.ones((1, self.n_neighbors))
                b = np.ones((1, 1))
                solution = solvers.qp(P = matrix(P), q = matrix(q), G = matrix(G),  
                                  h = matrix(h), A = matrix(A), b = matrix(b))
                weights = solution["x"]
                
            else:
                local_cov = np.cov(local_gram)
                
                trace = np.trace(local_cov)
                if trace > 0:
                    regularization = 0.001 * trace
                else:
                    regularization = 0.001
                
                local_cov.flat[::self.n_neighbors + 1] += regularization
                weights = np.linalg.solve(local_cov, np.ones(self.n_neighbors).T)
                weights = weights/weights.sum()
                
            for n in range(self.n_neighbors):
                self.W[i, neighbor_indices[n]] = weights[n]

def demo(seed = 0, per = 0.5, alpha = 0.99, max_iter = 1000, threshold = 0.001, 
                 n_neighbors = 7, allow_negative = False):
    iris = datasets.load_iris()
    data, labels = iris.data, iris.target
    rng = np.random.RandomState(seed)
    flag = rng.rand(len(data)) < per
    train_data, train_label = data[flag, :], labels[flag]
    test_data, test_target = data[~flag, :], labels[~flag]
    llp = LLP(alpha = alpha, max_iter = max_iter, threshold = threshold, 
                 n_neighbors = n_neighbors, allow_negative = allow_negative)
    llp.train(train_data, train_label)
    predict = llp.test(test_data)
    accu = np.where(test_target == predict)[0].shape[0] / len(test_target)
    print(accu)
    return accu

if __name__ == "__main__":
    # hard = []
    # soft = []
    # for i in range(100):
    #     soft.append(demo(seed = i, per = 0.5, alpha = 0.99))
    #     hard.append(demo(seed = i, per = 0.5, alpha = 1))
    # print("hard clamping, mean", np.mean(hard), ",std:", np.std(hard))
    # print("soft clamping, mean", np.mean(soft), ",std:", np.std(soft))

    for i in range(10):

        print("allow negative weights")
        demo(seed = i, per = 0.5, alpha = 0.99, max_iter = 50, threshold = 0.001, 
                           n_neighbors = 7, allow_negative = True)
        demo(seed = i, per = 0.5, alpha = 1, max_iter = 50, threshold = 0.001,
                           n_neighbors = 7, allow_negative = True)
        
        print("non-negative weights")
        demo(seed = i, per = 0.5, alpha = 0.99, max_iter = 50, threshold = 0.001, 
                           n_neighbors = 7, allow_negative = False)
    
        demo(seed = i, per = 0.5, alpha = 1, max_iter = 50, threshold = 0.001, 
                           n_neighbors = 7, allow_negative = False) 