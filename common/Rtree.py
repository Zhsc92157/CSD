# -*- coding:utf-8 -*-

import random
from heapq import heappop, heappush

from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import common


def bounding_box(geoms):
    return unary_union(geoms).envelope


class RtreeIndex(object):
    def __init__(self, minx, miny, maxx, maxy, data):
        self.count = 0
        self.leaf_count = 0
        self.MAX_CHILDREN_NUM = 10
        self.MIN_CHILDREN_NUM = int(self.MAX_CHILDREN_NUM / 2)
        self.CLUSTER_NUM = 2
        self.root = Node.create(self, None, Polygon(), [])
        self.space = box(minx, miny, maxx, maxy)
        self.geometries = dict()
        self.points = self.geometries
        for e in data:
            self.insert(e[0], e[1])

    def insert(self, o, geometry):
        root = self.root.insert(o, geometry)
        if len(root) > 1:
            self.root = Node.create_with_children(self, root)
        self.geometries[o] = geometry

    def get_all(self, node=None):
        if node is None:
            node = self.root
        yield node.obj, node.geom
        if not node.is_data_node:
            for c in node.children:
                for n in self.get_all(c):
                    yield n

    def range(self, r):
        for x in self.root.search(lambda x: r.intersects(x.geom), lambda x: r.intersects(x.geom)):
            yield x.obj, x.geom

    def nearest(self, q, k=1):
        min_heap = common.MinHeap()
        knn = common.MaxHeap()
        min_heap.push((0, self.root))
        while len(min_heap) > 0:
            node_min_dist, node = min_heap.pop()
            if len(knn) >= k and node_min_dist > knn.first()[0]:
                break
            if node.is_leaf_node:
                for c in node.children:
                    c_min_dist = c.geom.distance(q)
                    if len(knn) < k or c_min_dist < knn.first()[0]:
                        knn.push((c_min_dist, c))
                        if len(knn) > k:
                            knn.pop()
            else:
                for c in node.children:
                    c_min_dist = c.geom.distance(q)
                    if len(knn) < k or c_min_dist < knn.first()[0]:
                        min_heap.push((c_min_dist, c))
        return [(e[1].obj, e[1].geom, e[0]) for e in knn.values]


class Node(object):
    @classmethod
    def create(cls, tree, obj, geom, children):
        return Node(tree, obj, geom, children)

    @classmethod
    def create_with_children(cls, tree, children):
        geom = bounding_box([c.geom for c in children])
        node = Node.create(tree, None, geom, children)
        assert (not node.is_data_node)
        return node

    @classmethod
    def create_data_node(cls, tree, obj, geom):
        node = Node.create(tree, obj, geom, None)
        assert node.is_data_node
        return node

    def __init__(self, tree, obj, geom, children):
        self.tree = tree
        self.obj = obj
        self.geom = geom
        self.children = children

    def search(self, intermediate_node_predicate, data_node_predicate):
        if self.is_data_node:
            if data_node_predicate(self):
                yield self
        else:
            if intermediate_node_predicate(self):
                for c in self.children:
                    for cr in c.search(intermediate_node_predicate, data_node_predicate):
                        yield cr

    @property
    def is_data_node(self):
        if self.children is not None:
            return False
        if self.geom is not None and self.obj is not None:
            return True
        return False

    @property
    def is_leaf_node(self):
        if self.is_data_node:
            return False
        if len(self.children) == 0 or self.children[0].is_data_node:
            return True
        return False

    @property
    def is_intermediate_node(self):
        if (not self.is_data_node) and (not self.is_leaf_node):
            return True

    def insert(self, obj, geom):
        while True:
            if self.is_leaf_node:
                self.geom = bounding_box([self.geom, geom])
                self.children.append(Node.create_data_node(self.tree, obj, geom))
                return self.adjust()
            else:
                min_volume = -1
                insert_idx = None
                for i in range(len(self.children)):
                    volume = bounding_box([self.children[i].geom, geom]).area
                    if volume < min_volume or min_volume < 0:
                        min_volume = volume
                        insert_idx = i
                self.geom = bounding_box([self.geom, geom])
                inserting_child = self.children[insert_idx]
                inserting_child = inserting_child.insert(obj, geom)
                if len(inserting_child) > 1:
                    del self.children[insert_idx]
                    self.children += inserting_child
                return self.adjust()

    def adjust(self):
        if len(self.children) <= self.tree.MAX_CHILDREN_NUM:
            return [self]
        return self.split()



    def split(self):
        ExternalNode1, ExternalNode2, RemainNodes = self.pickseeds()
        threshold = self.tree.MAX_CHILDREN_NUM - self.tree.MIN_CHILDREN_NUM + 1
        while (len(ExternalNode1) != threshold and len(ExternalNode2) != threshold and len(RemainNodes) != 0):
            d1 = []
            d2 = []
            Node1Geom = [c.geom for c in ExternalNode1]
            Node2Geom = [d.geom for d in ExternalNode2]
            NodesGeom = [e.geom for e in RemainNodes]
            for i in range(len(NodesGeom)):
                d1.append(self.differenceArea(Node1Geom, NodesGeom[i]))
                d2.append(self.differenceArea(Node2Geom, NodesGeom[i]))
            index = self.getMaxDifferenceIndex(d1, d2)
            if d1[index] < d2[index]:
                ExternalNode1.append(RemainNodes[index])
            elif d1[index] > d2[index]:
                ExternalNode2.append(RemainNodes[index])
            else:
                if bounding_box(Node1Geom).area < bounding_box(Node2Geom).area:
                    ExternalNode1.append(RemainNodes[index])
                elif bounding_box(Node1Geom).area > bounding_box(Node2Geom).area:
                    ExternalNode2.append(RemainNodes[index])
                else:
                    if len(ExternalNode1) < len(ExternalNode2):
                        ExternalNode1.append(RemainNodes[index])
                    elif len(ExternalNode1) < len(ExternalNode2):
                        ExternalNode2.append(RemainNodes[index])
                    else:
                        ExternalNode1.append(RemainNodes[index])
            RemainNodes.pop(index)
        if len(ExternalNode1) == threshold:
            for node in RemainNodes:
                ExternalNode2.append(node)
        else:
            for node in RemainNodes:
                ExternalNode1.append(node)

        return [Node.create_with_children(self.tree, c) for c in [ExternalNode1, ExternalNode2]]

    def pickseeds(self):
        node1index = 0
        node2index = 0
        freeArea = 0
        for i in range(len(self.children)):
            for j in range(i + 1, len(self.children)):
                cndiateMBR = bounding_box([self.children[i].geom, self.children[j].geom]).area
                condiatefreearea = cndiateMBR - self.children[i].geom.area - self.children[j].geom.area
                if condiatefreearea > freeArea:
                    node1index = i
                    node2index = j
                    freeArea = condiatefreearea
        nodelist = []
        for i in range(len(self.children)):
            if i == node1index or i == node2index:
                pass
            else:
                nodelist.append(self.children[i])
        return [self.children[node1index]], [self.children[node2index]], nodelist

    def differenceArea(self, nodes, node):
        area1 = bounding_box(nodes).area
        nodes.append(node)
        area2 = bounding_box(nodes).area
        return abs(area1 - area2)

    def getMaxDifferenceIndex(self, d1, d2):
        maxDiff = 0
        index = 0
        for i in range(len(d1)):
            Diff = abs(d1[i] - d2[i])
            if Diff > maxDiff:
                maxDiff = Diff
                index = i
        return index

    # def split(self):
    #     clusters = k_means_cluster(self.tree.CLUSTER_NUM, self.children)
    #     return [Node.create_with_children(self.tree, c) for c in clusters]


def k_means_cluster(k, nodes):
    if len(nodes) <= k:
        return [[node] for node in nodes]
    centers = [node.geom.centroid.coords[0] for node in nodes]
    clustering = KMeans(n_clusters=k)
    clustering.fit(centers)
    labels = clustering.labels_
    return [[nodes[j] for j in range(len(nodes)) if labels[j] == i] for i in range(k)]


def plot_geometry(ax, geometry, color='gray'):
    if isinstance(geometry, Point):
        ax.plot(geometry.x, geometry.y, '.', color='red')
    else:
        coordinates = list(geometry.exterior.coords)
        ax.plot([c[0] for c in coordinates], [c[1] for c in coordinates], '-', color=color)
