### Code
    Improved Graph Convolution Network with Feature Encoding

### Improved Graph Convolution Network
    1. First Order Chebyshev Polynomial Approximation for Spectral-Based Graph Convolution Network
    2. Faster training, no need for expensive Eigen Decomposition

### Encoding Methods
    1. Extend semi-supervised GCN to supervised learning framework by encoding unseen testing sample back to the training manifold
    2. Replacement: do not add/delete edges, only modify the weight
    3. Embedding: modify the weight and add/delete edges as well

### Label Propagation
    1. Global Label Propagation using global Gram matrix / global pairwise shortest path matrix / global normalized graph Laplacian matrix as propagation matrix  
    2. Local Label Propagation using LLE embedding matrix or local reconstruction matrix based on Local Gram matrix as propagation matrix  
    3. Label clamping : Soft and Hard clamping 

### Temporal Mapping
    1. Linear PixMat and RandMat
    2. Nonlinear PixMat and RandMat