# CSD: Discriminance with Conic Section for Improving Reverse *k* Nearest Neighbors Queries
The reverse *k* nearest neighbors (R*k*NN) requires to find every data point that has the query point as one of its *k* closest points. 
According to the characteristics of conic section, we propose a discriminance, named CSD (Conic Section Discriminance), to determine candidates whether belong to the R*k*NN set or not.
With CSD, the vast majority of candidates can be verified with a computational complexity of *O(1)*.
Based on CSD, a novel R*k*NN algorithm CSD-R*k*NN is implemented.
The comparative experiments are  conducted between CSD-R*k*NN and other two state-of-the-art R*k*NN algorithms, SLICE and VR-R*k*NN.
The experimental results indicate that the efficiency of CSD-R*k*NN is significantly higher than the other two algorithms.
