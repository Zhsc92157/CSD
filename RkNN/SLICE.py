from math import cos, pi, sin, acos
from time import time

from numpy.linalg import norm
from shapely.geometry import Point
import numpy as np

import common


def MonoRkNN(q_id, k, index, partition_num=12):
    begin_time = time()
    q = index.geometries[q_id]
    sigLists, unpruned_area, pruning_io = pruning(q_id, q, k + 1, index, partition_num)
    pruning_io += 1
    pruning_time = time()
    result, verification_io, candidate_num = mono_verification(q, k, sigLists, unpruned_area)
    verification_time = time()
    return result, pruning_time - begin_time, verification_time - pruning_time, pruning_io, verification_io, candidate_num, candidate_num


def mono_verification(q, k, sigLists, unpruned_area):
    result = list()
    candidates, IO = mono_retrieve_candidates(sigLists, unpruned_area)
    sigLists = [sorted(s, key=lambda c: c[0]) for s in sigLists]
    for candidate in candidates:
        if mono_is_RkNN(q, candidate[0], candidate[1], k, sigLists):
            result.append(candidate)
    return result, IO, len(candidates)


def mono_is_RkNN(q, candidate_id, candidate, k, sigLists):
    i = sector_id(candidate, q, len(sigLists))
    count = 0
    sigList = sigLists[i]
    for r_l, f_id, f in sigList:
        if candidate_id == f_id:
            continue
        if candidate.distance(q) < r_l:
            return True
        if candidate.distance(f) < candidate.distance(q):
            count += 1
            if count >= k:
                return False
    return True


def mono_retrieve_candidates(sigLists, unpruned_area):
    candidates = list()
    visited = set()
    for sigList in sigLists:
        for r_l, f_id, f in sigList:
            if f_id not in visited:
                visited.add(f_id)
                if unpruned_area.intersects(f):
                    candidates.append((f_id, f))
    return candidates, 0


def BiRkNN(q_id, k, facility_index, user_index, partition_num=12):
    begin_time = time()
    q = facility_index.geometries[q_id]
    sigLists, unpruned_area, pruning_io = pruning(q_id, q, k, facility_index, partition_num)
    pruning_io += 1
    pruning_time = time()
    result, verification_io, candidate_num = bi_verification(q, k, user_index, sigLists, unpruned_area)
    verification_time = time()
    return result, pruning_time - begin_time, verification_time - pruning_time, pruning_io, verification_io, candidate_num, candidate_num


def pruning(q_id, q, k, index, partition_num):
    partitions =common.partitions(q, index.space, partition_num)
    sigLists = [[] for i in range(partition_num)]
    upper_arc_radius_heaps = [common.MaxHeap() for i in range(partition_num)]
    shaded_areas = [calculate_shaded_area(partition, partition.r) for partition in partitions]
    h = common.MinHeap()
    IO = 0
    h.push((0, index.root))
    while len(h) > 0:
        e_dist, e = h.pop()
        if may_contains_significant_facility(e, shaded_areas):
            if e.is_data_node:
                pruneSpace(q_id, e, k, partitions, sigLists, upper_arc_radius_heaps, shaded_areas)
            else:
                for child in e.children:
                    h.push((child.geom.distance(q), child))
                    IO += 1

    unpruned_area_list = list()
    for i in range(partition_num):
        r_b = min(upper_arc_radius_heaps[i].first(), partitions[i].r)
        angles = [2 * pi / partition_num * i, 2 * pi / partition_num * (i + 1)]
        if r_b > 0:
            unpruned_area_list.append(common.sector(q, r_b, angles).buffer(0.01))
    unpruned_area = reduce(lambda x, y: x.union(y), unpruned_area_list)

    return sigLists, unpruned_area, IO


def pruneSpace(q_id, e, k, partitions, sigLists, upper_arc_radius_heaps, shaded_areas):
    f_id = e.obj
    if f_id == q_id:
        return
    f = e.geom
    for i in range(len(partitions)):
        partition = partitions[i]
        sigList = sigLists[i]
        upper_arc_radius_heap = upper_arc_radius_heaps[i]
        min_angle, max_angle = min_and_max_angle(f, partition)
        if min_angle < pi / 2:
            r_l, r_u = lower_and_upper_arc_radius(f, partition)
            bounding_arc_radius = float('inf')
            if len(upper_arc_radius_heap) < k or r_u < upper_arc_radius_heap.first():
                upper_arc_radius_heap.push(r_u)
                if len(upper_arc_radius_heap) > k:
                    upper_arc_radius_heap.pop()
                if len(upper_arc_radius_heap) == k:
                    bounding_arc_radius = upper_arc_radius_heap.first()

                    shaded_areas[i] = calculate_shaded_area(partition, bounding_arc_radius)
            if is_significant_facility(f, partition, bounding_arc_radius):
                sigList.append([r_l, f_id, f])


def bi_verification(q, k, index, sigLists, unpruned_area):
    result = list()
    candidates, IO = bi_retrieve_candidates(index, unpruned_area)
    sigLists = [sorted(s, key=lambda c: c[0]) for s in sigLists]
    for candidate in candidates:
        if bi_is_RkNN(q, candidate[1], k, sigLists):
            result.append(candidate)
    return result, IO, len(candidates)


def bi_retrieve_candidates(index, unpruned_area):
    candidates = list()
    IO = 0
    entries = {index.root}
    IO += 1
    while len(entries) > 0:
        e = entries.pop()
        if unpruned_area.intersects(e.geom):
            if e.is_data_node:
                candidates.append((e.obj, e.geom))
            else:
                for child in e.children:
                    entries.add(child)
                    IO += 1
    return candidates, IO


def bi_is_RkNN(q, u, k, sigLists):
    i = sector_id(u, q, len(sigLists))
    count = 0
    sigList = sigLists[i]
    for r_l, f_id, f in sigList:
        if u.distance(q) < r_l:
            return True
        if u.distance(f) < u.distance(q):
            count += 1
            if count >= k:
                return False
    return True


def is_significant_facility(f, partition, bounding_arc_radius):
    if partition.contains(f):
        if f.distance(partition.o) > 2 * bounding_arc_radius:
            return False
    else:
        M, N = get_M_N(partition, bounding_arc_radius)
        if M.distance(f) > bounding_arc_radius and N.distance(f) > bounding_arc_radius:
            return False
    return True


def may_contains_significant_facility(e, shaded_areas):
    for area in shaded_areas:
        if area.intersects(e.geom):
            return True
    return False


def calculate_shaded_area(partition, bounding_arc_radius):
    if bounding_arc_radius == 0:
        return partition.origin
    if bounding_arc_radius == float('inf'):
        bounding_arc_radius = partition.r
    sector = common.sector(partition.o, bounding_arc_radius * 2, partition.angles)
    m, n = get_M_N(partition, bounding_arc_radius)
    circle_m = common.circle(m, bounding_arc_radius)
    circle_n = common.circle(n, bounding_arc_radius)
    return sector.union(circle_m).union(circle_n)


def get_M_N(partition, bounding_arc_radius):
    o = partition.o
    if bounding_arc_radius == float('inf'):
        l = partition.r
    else:
        l = bounding_arc_radius
    angles = partition.angles
    M = Point(cos(angles[0]) * l + o.x, sin(angles[0]) * l + o.y)
    N = Point(cos(angles[1]) * l + o.x, sin(angles[1]) * l + o.y)
    return M, N


def min_and_max_angle(f, partition):
    o = (partition.o.x, partition.o.y)
    p1 = [o[0] + cos(partition.angles[0]), o[1] + sin(partition.angles[0])]
    p2 = [o[0] + cos(partition.angles[1]), o[1] + sin(partition.angles[1])]
    angles = [angle(o, p1, f), angle(o, p2, f)]
    return sorted(angles)


def lower_and_upper_arc_radius(f, partition):
    min_angle, max_angle = min_and_max_angle(f, partition)
    dist_f_o = f.distance(partition.o)
    if max_angle >= pi / 2:
        r_u = float('inf')
    else:
        r_u = dist_f_o / (2 * cos(max_angle))
    r_l = dist_f_o / (2 * cos(min_angle))
    return r_l, r_u


def angle(o, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    o = np.asarray(o)
    vector_ox = x - o
    vector_oy = y - o
    norm_ox = norm(vector_ox)
    norm_oy = norm(vector_oy)
    if norm_ox == 0 or norm_oy == 0:
        return 0
    return acos(vector_ox.dot(vector_oy) / (norm_ox * norm_oy))


def sector_id(p, origin, sector_num):
    return int(angle_with_x(origin, p) / (2 * pi / sector_num))

def angle_with_x(p_start, p_end):
    dist = p_start.distance(p_end)
    if dist == 0:
        return 0
    if p_end.y - p_start.y >= 0:
        return acos((p_end.x - p_start.x) / dist)
    else:
        return 2 * pi - acos((p_end.x - p_start.x) / dist)
