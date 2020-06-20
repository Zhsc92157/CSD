from heapq import heappop, heappush, heappushpop, heapify, _heapify_max, _heappushpop_max, _siftdown_max, _siftup_max

from collections import Iterable
from math import cos, ceil, pi, sin

from shapely.geometry import Polygon, Point


def circle(o, r, resolution=None):
    if r <= 0:
        raise ValueError("r must be a number greater than 0")
    if resolution:
        return o.buffer(r, int(ceil(pi * r * 2 / resolution / 4)))
    else:
        return o.buffer(r, 32)


def sector(o, r, angles, resolution=None):
    c = circle(o, r, resolution)
    if abs(angles[0] - angles[1]) >= pi:
        raise ValueError('abs(angles[0] - angles[1]) must be less than Pi')
    l = r / cos(abs(angles[0] - angles[1]) / 2)
    triangle = Polygon(
        [(o.x, o.y), (o.x + cos(angles[0]) * l, o.y + sin(angles[0]) * l),
         (o.x + cos(angles[1]) * l, o.y + sin(angles[1]) * l)])
    s = triangle.intersection(c)
    s.o = o
    s.r = r
    s.angles = angles
    return s


def partitions(origin, space, n):
    bounds = space.bounds

    r = Point((bounds[0], bounds[1])).distance(Point((bounds[2], bounds[3])))
    return [sector(origin, r, [2 * pi / n * i, 2 * pi / n * (i + 1)]) for i in range(n)]


def heappush_max(heap, item):
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)


def heappop_max(heap):
    last = heap.pop()
    if heap:
        return_item = heap[0]
        heap[0] = last
        _siftup_max(heap, 0)
    else:
        return_item = last
    return return_item


class MinHeap(Iterable):
    def __init__(self):
        self.items = []

    def pop(self):
        return heappop(self.items)

    def push(self, item):
        heappush(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    smallest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i


class MaxHeap(Iterable):
    def __init__(self):
        self.items = []

    def pop(self):
        return heappop_max(self.items)

    def push(self, item):
        heappush_max(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    largest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i


class NSmallestHolder:
    def __init__(self, n):
        self.items = []
        self.n = n

    def push(self, item):
        if len(self.items) < self.n:
            self.items.append(item)
            if len(self.items) == self.n:
                _heapify_max(self.items)
        else:
            _heappushpop_max(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    largest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i


class NLargestHolder:
    def __init__(self, n):
        self.items = []
        self.n = n

    def push(self, item):
        if len(self.items) < self.n:
            self.items.append(item)
            if len(self.items) == self.n:
                heapify(self.items)
        else:
            heappushpop(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    smallest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i
