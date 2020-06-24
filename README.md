# CSD: Discriminance with Conic Section for Improving Reverse *k* Nearest Neighbors Queries
## Overview
The reverse *k* nearest neighbors (R*k*NN) requires to find every data point that has the query point as one of its *k* closest points. 
According to the characteristics of conic section, we propose a discriminance, named CSD (Conic Section Discriminance), to determine candidates whether belong to the R*k*NN set or not.
With CSD, the vast majority of candidates can be verified with a computational complexity of *O(1)*.
Based on CSD, a novel R*k*NN algorithm CSD-R*k*NN is implemented.
The comparative experiments are  conducted between CSD-R*k*NN and other two state-of-the-art R*k*NN algorithms, SLICE and VR-R*k*NN.
The experimental results indicate that the efficiency of CSD-R*k*NN is significantly higher than the other two algorithms.
## Project structure
```
├── data/: real data set
│   └── us50000.txt
├── common/: common data structures (including Min-heap, Max-heap, Voronoi dagram, Rtree, KDtree, VoRtree and VoKDtree)
│   ├── __init__.py
│   ├── VoronoiDiagram.py
│   ├── Rtree.py
│   ├── KDtree.py
│   ├── VoRtree.py
│   └── VoKDtree.py
├── RkNN/: RkNN algorithms (including CSD-RkNN, SLICE and VR-RkNN)
│   ├── __init__.py
│   ├── CSD.py
│   ├── SLICE.py
│   └── VR.py
└── test/: test benchmarks
    └── benchmark.py
```
## Usage
```python
TBD
```
