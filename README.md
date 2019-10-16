### Overview of activities since 2019 Spring Meeting

### Improved Graph Convolution Network
    1. First Order Chebyshev Polynomial Approximation for Spectral-Based Graph Convolution Network
    2. Faster training, no need for expensive Eigen Decomposition

### Encoding Methods
    1. Different than literature by extending semi-supervised GCN to supervised learning framework by encoding unseen testing sample back to the training manifold

    2. Replacement: do not add/delete edges, only modify the weight
    3. Embedding: modify the weight and add/delete edges as well

### Label Propagation
    1. Global Label Propagation using global Gram matrix / global pairwise shortest path matrix / global normalized graph Laplacian matrix as propagation matrix  
    2. Local Label Propagation using LLE embedding matrix or local reconstruction matrix based on Local Gram matrix as propagation matrix  
    3. Label clamping : Soft and Hard clamping 

### Temporal Mapping
    1. Linear PixMat and RandMat
    2. Nonlinear PixMat and RandMat
    
### Data
    1. For GCN / Label Propagation:  
        aggregated 2013 Spring polygon-level feature 
    2. For Temporal Mapping:
        AVIRIS spectra and metadata file for 2013 Spring and 2014 Spring
    3. SusanSpectraProcessed130411_classesremoved.mat contains pixel-level features