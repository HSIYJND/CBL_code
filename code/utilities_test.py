import unittest
from utilities import *
import numpy as np

class TestUnits(unittest.TestCase):

    def test_graph_fourier_transform(self):
        A = np.random.rand(10,10)
        A = A + A.T
        U1, S1, info1 = graph_fourier_transform(A, normalized = "symmetric")
        U2, S2, info2 = graph_fourier_transform(A, normalized = "unnormalized")
        U3, S3, info3 = graph_fourier_transform(A, normalized = "random")

    def test_aggregate_using_unweighted_mean(self):
        data = np.random.rand(10,5)
        data = aggregate_using_unweighted_mean(data)
        self.assertEqual(data.shape[0], 1)
        self.assertEqual(data.shape[1], 5)
    
    def test_aggregate_using_weighted_mean(self):
        data = np.random.rand(10,5)
        data = aggregate_using_weighted_mean(data)
        self.assertEqual(data.shape[0], 1)
        self.assertEqual(data.shape[1], 5)

    def test_aggregate_using_truncated_mean(self):
        data = np.random.rand(10,5)
        data = aggregate_using_truncated_mean(data, alpha = 0.1)
        self.assertEqual(data.shape[0], 1)
        self.assertEqual(data.shape[1], 5)
    
    def test_polygon_aggregation_multi_source(self):
        sensor_data_1 = np.random.rand(10,5)
        sensor_data_2 = np.random.rand(10,3)
        sensor_data_3 = np.random.rand(10,2)
        data = polygon_aggregation([sensor_data_1, sensor_data_2, sensor_data_3], aggregate_using_unweighted_mean)
        self.assertEqual(len(data), 3)
        self.assertTrue(np.all(data[0] == aggregate_using_unweighted_mean(sensor_data_1)))
        self.assertTrue(np.all(data[1] == aggregate_using_unweighted_mean(sensor_data_2)))
        self.assertTrue(np.all(data[2] == aggregate_using_unweighted_mean(sensor_data_3)))
    
    def test_polygon_aggregation_single_source(self):
        sensor_data = np.random.rand(10,5)
        data = polygon_aggregation(sensor_data, aggregate_using_weighted_mean)
        self.assertTrue(np.all(data[0] == aggregate_using_weighted_mean(sensor_data)))
        self.assertEqual(data.shape[0], 1)
        self.assertEqual(data.shape[1], sensor_data.shape[1])
    
    def test_construct_adjacency_matrix_fixed_size(self):
        dist = np.random.rand(100,100)
        dist += dist.T
        adj = construct_adjacency_matrix(dist, method = "fixed", n_neighbors = 20)
        for i in range(adj.shape[0]):
            self.assertEqual(np.where(adj[i] != 0)[0].shape[0], 20)
    
    def test_construct_adjacency_matrix_radius_size(self):
        dist = np.random.rand(100,100)
        dist += dist.T
        adj = construct_adjacency_matrix(dist, method = "radius", sigma = 0.5)
        self.assertTrue(np.all(adj == adj.T))

if __name__ == "__main__":
    unittest.main()