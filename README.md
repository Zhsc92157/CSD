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
Generate the facility set and user set:
```python
>>> import numpy as np
>>> from shapely.geometry import Point
>>> bounds = (0, 0, 10000, 10000)
>>> users = [(i, Point(np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3]))) for i in
             range(1000)]
>>> facilities = [(i, Point(np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3]))) for i in
                  range(1000)]
```
Index the facilities and users:
```python
>>> from common.VoKDtree import VoKDtreeIndex
>>> user_index = VoKDtreeIndex(bounds[0], bounds[1], bounds[2], bounds[3], users)
>>> facility_index = VoKDtreeIndex(bounds[0], bounds[1], bounds[2], bounds[3], facilities)
```
Retrieve the R*k*NNs of the (*q*-1)th facility form the user set:
```python
>>> from RkNN.CSD import BiRkNN
>>> q, k = np.random.randint(0, len(facilities)), 10 
>>> print BiRkNN(q, k, facility_index, user_index)
[(937, <shapely.geometry.point.Point object at 0x11731ec90>), (367, <shapely.geometry.point.Point object at 0x1172e2bd0>), 
 (155, <shapely.geometry.point.Point object at 0x117281610>), (143, <shapely.geometry.point.Point object at 0x117281310>), 
 (52, <shapely.geometry.point.Point object at 0x117276bd0>), (965, <shapely.geometry.point.Point object at 0x1173243d0>), 
 (730, <shapely.geometry.point.Point object at 0x11730a810>), (720, <shapely.geometry.point.Point object at 0x11730a590>), 
 (306, <shapely.geometry.point.Point object at 0x1172dcc50>), (915, <shapely.geometry.point.Point object at 0x11731e710>)]
```
