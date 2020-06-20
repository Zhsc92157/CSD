from scipy.spatial import KDTree as ScipyKDTree
from shapely.geometry import Point, box

import common
from common.KDtree import KDTree
from common.VoronoiDiagram import VoronoiDiagram
import matplotlib.pyplot as plt


class VoKDtreeIndex(KDTree):
    def __init__(self, minx, miny, maxx, maxy, data, leafsize=10):
        super(VoKDtreeIndex, self).__init__(minx, miny, maxx, maxy, data, leafsize)
        self.voronoi_diagram = VoronoiDiagram(data, (minx, miny), (maxx, maxy))

    def kNN(self, q, k):
        IO = 0
        knn = list()
        h = common.MinHeap()
        (nn_o, nn_p, nn_dist), nn_io = self.NN(q)
        IO += nn_io
        vd = self.voronoi_diagram
        points = self.points
        h.push((nn_dist, nn_o))
        visited = {nn_o}
        count = 0
        while count < k and len(h) > 0:
            dist, o = h.pop()
            p = points[o]
            knn.append((o, p, dist))
            for neighbor in vd.neighbors(o):
                if neighbor not in visited:
                    IO += 1
                    visited.add(neighbor)
                    h.push((points[neighbor].distance(q), neighbor))
            count += 1
        return knn, IO

    def nearest(self, q, k=1):
        if k == 1:
            return self.NN(q)
        else:
            return self.kNN(q, k)


class VoKDtreeIndex_Scipy:
    def __init__(self, minx, miny, maxx, maxy, data):
        self.data = data
        self.points = {e[0]: e[1] for e in data}
        self.space = box(minx, miny, maxx, maxy)
        self.kdtree = ScipyKDTree([(e[1].x, e[1].y) for e in data], leafsize=10)
        self.voronoi_diagram = VoronoiDiagram(data, (minx, miny), (maxx, maxy))

    def NN(self, q):
        dist, i = self.kdtree.query(q, k=1)
        o, p = self.data[i]
        return o, p, dist

    def kNN(self, q, k):
        knn = list()
        h = common.MinHeap()
        nn_o, nn_p, nn_dist = self.NN(q)
        vd = self.voronoi_diagram
        points = self.points
        h.push((nn_dist, nn_o))
        visited = {nn_o}
        count = 0
        while count < k and len(h) > 0:
            dist, o = h.pop()
            p = points[o]
            knn.append((o, p, dist))
            for neighbor in vd.neighbors(o):
                if neighbor not in visited:
                    visited.add(neighbor)
                    h.push((points[neighbor].distance(q), neighbor))
            count += 1
        return knn

    def nearest(self, q, k=1):
        if k == 1:
            return self.NN(q)
        else:
            return self.kNN(q, k)
