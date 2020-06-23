import sys
import numpy as np
from heapq import heappush, heappop
from scipy.spatial import minkowski_distance
from shapely.geometry import box, Point
import matplotlib.pyplot as plt

import common


class KDTree(object):
    def __init__(self, minx, miny, maxx, maxy, data, leafsize=10):
        self.data = data
        self.points = {e[0]: e[1] for e in data}
        self.idx = [e[0] for e in data]
        self.data_size = len(data)
        self.space = box(minx, miny, maxx, maxy)
        self.dimension = len(data[0][1].coords[0])
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.asarray([maxx, maxy])
        self.mins = np.asarray([minx, miny])
        self.root = self.__build(self.idx, self.maxes, self.mins)

    class Node(object):
        pass

    class LeafNode(Node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class IntermediateNode(Node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children + greater.children

    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            node = KDTree.LeafNode(idx)
            node.box = box(mins[0], mins[1], maxes[0], maxes[1])
            return node
        else:
            d = np.argmax(maxes - mins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval == minval:
                node = KDTree.LeafNode(idx)
                node.box = box(mins[0], mins[1], maxes[0], maxes[1])
                return node
            data = np.asarray([self.points[i].coords[0][d] for i in idx])

            split = (maxval + minval) / 2
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
            if len(less_idx) == 0:
                split = np.amin(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
            if len(greater_idx) == 0:
                split = np.amax(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
            if len(less_idx) == 0:
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                less_idx = np.arange(len(data) - 1)
                greater_idx = np.array([len(data) - 1])

            lessmaxes = np.copy(maxes)
            lessmaxes[d] = split
            greatermins = np.copy(mins)
            greatermins[d] = split
            node = KDTree.IntermediateNode(d, split,
                                           self.__build([idx[i] for i in less_idx], lessmaxes, mins),
                                           self.__build([idx[i] for i in greater_idx], maxes, greatermins))
            node.box = box(mins[0], mins[1], maxes[0], maxes[1])
            return node

    def kNN(self, q, k):
        min_distance = q.distance(self.root.box)
        heap = common.MinHeap()
        heap.push((min_distance, self.root))
        neighbors = []
        distance_upper_bound = np.inf
        while len(heap) > 0:
            min_distance, node = heap.pop()
            if isinstance(node, KDTree.LeafNode):
                for i in node.idx:
                    p = self.points[i]
                    dist = p.distance(q)
                    if dist < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-dist, i))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                if min_distance > distance_upper_bound:
                    break
                if q.coords[0][node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less
                heap.push((min_distance, near))

                far_min_distance = far.box.distance(q)

                if far_min_distance <= distance_upper_bound:
                    heap.push((far_min_distance, far))
        return sorted([(i, self.points[i], -d) for (d, i) in neighbors])

    def NN(self, q):
        IO = 0
        root = self.root
        IO += 1
        heap = common.MinHeap()
        heap.push((q.distance(root.box), root))
        distance_upper_bound = np.inf
        nn_id = None
        while len(heap) > 0:
            min_distance, node = heap.pop()
            if isinstance(node, KDTree.LeafNode):
                for i in node.idx:
                    IO += 1
                    p = self.points[i]
                    dist = p.distance(q)
                    if dist < distance_upper_bound:
                        nn_id = i
                        distance_upper_bound = dist
            else:
                if min_distance > distance_upper_bound:
                    break
                if q.coords[0][node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less
                IO += 2
                heap.push((min_distance, near))
                far_min_distance = far.box.distance(q)
                if far_min_distance <= distance_upper_bound:
                    heap.push((far_min_distance, far))
        return (nn_id, self.points[nn_id], distance_upper_bound), IO
