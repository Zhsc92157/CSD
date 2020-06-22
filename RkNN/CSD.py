from time import time

import common
from common import MinHeap


def MonoRkNN(q_id, k, index):
    begin_time = time()
    cache = dict()
    q = index.points[q_id]
    cache[q_id] = q
    candidates, pruning_io = mono_pruning(q_id, q, k, index, cache)
    pruning_io += 1
    pruning_time = time()
    result, verification_io, candidate_num, verified_candidate_num = mono_verification(q_id, q, k, candidates, index,
                                                                                       cache)
    verification_time = time()
    return result, pruning_time - begin_time, verification_time - pruning_time, pruning_io, verification_io, candidate_num, verified_candidate_num


def mono_pruning(q_id, q, k, index, cache):
    IO = 0
    partitions = list(common.partitions(q, index.space, 6))
    points = index.points
    vd = index.voronoi_diagram
    candidate_idx = set()
    candidates = list()
    for partition in partitions:
        h = MinHeap()
        h.push((0, q_id, q))
        visited = {q_id}
        knn = list()
        while len(h) > 0 and len(knn) < k:
            dist, o, p = h.pop()
            if o != q_id and partition.intersects(p):
                knn.append((o, p, dist))
                if o not in candidate_idx:
                    candidate_idx.add(o)
                    candidates.append((o, p, dist))
            for neighbor in vd.neighbors(o):
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in cache:
                        neighbor_p = cache[neighbor]
                    else:
                        neighbor_p = points[neighbor]
                        cache[neighbor] = neighbor_p
                        IO += 1
                    if partition.intersects(vd.cell(neighbor)):
                        h.push((neighbor_p.distance(q), neighbor, neighbor_p))
    return candidates, IO


def mono_verification(q_id, q, k, candidates, index, cache):
    IO = 0
    verified_candidate_num = 0
    candidates.sort(key=lambda x: x[2])
    r_q, r_q_io = mono_knn_radius(q_id, q, k, index, cache)
    semi_r_q = r_q / 2
    inner_candidates = list()
    outer_candidates = list()
    for c in candidates:
        if c[2] <= r_q:
            inner_candidates.append(c)
        else:
            outer_candidates.append(c)
    discriminant_dict = dict()
    rknn_idx = set()
    rknn = list()
    for i in range(len(inner_candidates) - 1, -1, -1):
        is_rknn, csd_io, is_verified = mono_CSD(inner_candidates[i], k, semi_r_q, rknn_idx, discriminant_dict, index,
                                                cache)
        IO += csd_io
        if is_verified:
            verified_candidate_num += 1
        if is_rknn:
            rknn_idx.add(inner_candidates[i][0])
            rknn.append((inner_candidates[i][0], inner_candidates[i][1]))
    for i in range(len(outer_candidates)):
        is_rknn, csd_io, is_verified = mono_CSD(outer_candidates[i], k, semi_r_q, rknn_idx, discriminant_dict, index,
                                                cache)
        IO += csd_io
        if is_verified:
            verified_candidate_num += 1
        if is_rknn:
            rknn_idx.add(outer_candidates[i][0])
            rknn.append((outer_candidates[i][0], outer_candidates[i][1]))

    return rknn, IO, len(candidates), verified_candidate_num


def mono_CSD(candidate, k, semi_r_q, RkNNs, discriminant_dict, index, cache):
    IO = 0
    p_distance = candidate[2]
    '''---------------------------------Semi-R Lemma---------------------------------'''
    if p_distance <= semi_r_q:
        return True, IO, False
    '''------------------------------------------------------------------------------'''
    p_id = candidate[0]
    p = candidate[1]
    '''--------------------------Ellipse and Hyperbola Lemma-------------------------'''
    vd = index.voronoi_diagram
    for neighbor in vd.neighbors(p_id):
        if neighbor in discriminant_dict:
            discriminant_point, r_pd = discriminant_dict[neighbor]
            if neighbor in RkNNs and p_distance + p.distance(discriminant_point) <= r_pd:
                discriminant_dict[p_id] = (discriminant_point, r_pd)
                return True, IO, False
            if neighbor not in RkNNs and p_distance - p.distance(discriminant_point) > r_pd:
                discriminant_dict[p_id] = (discriminant_point, r_pd)
                return False, IO, False
    '''------------------------------------------------------------------------------'''

    '''----------------------------------kNN-R Lemma---------------------------------'''
    r, IO = mono_knn_radius(p_id, p, k + 1, index, cache)
    discriminant_dict[p_id] = (p, r)
    if p_distance <= r:
        return True, IO, True
    else:
        return False, IO, True
    '''------------------------------------------------------------------------------'''


def mono_knn_radius(q_id, q, k, index, cache):
    knn, knn_io = kNN(index, q, k + 1, cache, (q_id, q, 0))
    return knn[-1][2], knn_io


def BiRkNN(q_id, k, facility_index, user_index):
    begin_time = time()
    facility_cache = dict()
    q = facility_index.points[q_id]
    facility_cache[q_id] = q
    candidate_region, pruning_io = bi_pruning(q_id, q, k, facility_index, facility_cache)
    pruning_io += 1
    pruning_time = time()
    result, verification_io, candidate_num, verified_candidate_num = bi_verification(q, k, candidate_region,
                                                                                     facility_index, user_index,
                                                                                     facility_cache)
    verification_time = time()
    return result, pruning_time - begin_time, verification_time - pruning_time, pruning_io, verification_io, candidate_num, verified_candidate_num


def bi_pruning(q_id, q, k, index, cache):
    IO = 0
    partitions = list(common.partitions(q, index.space, 6))
    region_list = list()
    points = index.points
    vd = index.voronoi_diagram
    for partition in partitions:
        h = MinHeap()
        h.push((0, q_id, q))
        visited = {q_id}
        knn = list()
        while len(h) > 0 and len(knn) < k:
            dist, o, p = h.pop()
            if o != q_id and partition.intersects(p):
                knn.append((o, p, dist))
            for neighbor in vd.neighbors(o):
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in cache:
                        neighbor_p = cache[neighbor]
                    else:
                        neighbor_p = points[neighbor]
                        cache[neighbor] = neighbor_p
                        IO += 1
                    if partition.intersects(vd.cell(neighbor)):
                        h.push((neighbor_p.distance(q), neighbor, neighbor_p))
        if len(knn) > 0:
            r = knn[-1][2]
            if r > 0:
                region_list.append(common.sector(q, r, partition.angles).buffer(0.01))
    region = reduce(lambda x, y: x.union(y), region_list)
    return region, IO


def bi_verification(q, k, candidate_region, facility_index, user_index, facility_cache):
    IO = 0
    verified_candidate_num = 0
    candidates, retrieve_candidates_io = bi_retrieve_candidates(q, user_index, candidate_region)
    IO += retrieve_candidates_io
    candidates = [(o, p, p.distance(q)) for o, p in candidates]
    candidates.sort(key=lambda x: x[2])
    r_q, r_q_io = bi_knn_radius(q, k - 1, facility_index, facility_cache)
    semi_r_q = r_q / 2
    inner_candidates = list()
    outer_candidates = list()
    for c in candidates:
        if c[2] <= r_q:
            inner_candidates.append(c)
        else:
            outer_candidates.append(c)
    discriminant_dict = dict()
    rknn_idx = set()
    rknn = list()
    for i in range(len(inner_candidates) - 1, -1, -1):
        is_rknn, csd_io, is_verified = bi_CSD(inner_candidates[i], k, semi_r_q, rknn_idx, discriminant_dict,
                                              facility_index,
                                              user_index,
                                              facility_cache)
        IO += csd_io
        if is_verified:
            verified_candidate_num += 1
        if is_rknn:
            rknn_idx.add(inner_candidates[i][0])
            rknn.append((inner_candidates[i][0], inner_candidates[i][1]))
    for i in range(len(outer_candidates)):
        is_rknn, csd_io, is_verified = bi_CSD(outer_candidates[i], k, semi_r_q, rknn_idx, discriminant_dict,
                                              facility_index,
                                              user_index,
                                              facility_cache)
        IO += csd_io
        if is_verified:
            verified_candidate_num += 1
        if is_rknn:
            rknn_idx.add(outer_candidates[i][0])
            rknn.append((outer_candidates[i][0], outer_candidates[i][1]))

    return rknn, IO, len(candidates), verified_candidate_num


def bi_CSD(candidate, k, semi_r_q, RkNNs, discriminant_dict, facility_index, user_index, cache):
    IO = 0
    p_distance = candidate[2]
    '''---------------------------------Semi-R Lemma---------------------------------'''
    if p_distance <= semi_r_q:
        return True, IO, False
    '''------------------------------------------------------------------------------'''
    p_id = candidate[0]
    p = candidate[1]
    '''--------------------------Ellipse and Hyperbola Lemma-------------------------'''
    vd = user_index.voronoi_diagram
    for neighbor in vd.neighbors(p_id):
        if neighbor in discriminant_dict:
            discriminant_point, r_pd = discriminant_dict[neighbor]
            if neighbor in RkNNs and p_distance + p.distance(discriminant_point) <= r_pd:
                discriminant_dict[p_id] = (discriminant_point, r_pd)
                return True, IO, False
            if neighbor not in RkNNs and p_distance - p.distance(discriminant_point) > r_pd:
                discriminant_dict[p_id] = (discriminant_point, r_pd)
                return False, IO, False
    '''------------------------------------------------------------------------------'''

    '''----------------------------------kNN-R Lemma---------------------------------'''
    r_p, IO = bi_knn_radius(p, k, facility_index, cache)
    discriminant_dict[p_id] = (p, r_p)
    if p_distance <= r_p:
        return True, IO, True
    else:
        return False, IO, True
    '''------------------------------------------------------------------------------'''


def bi_knn_radius(q, k, index, cache):
    knn, knn_io = kNN(index, q, k, cache)
    return knn[-1][2], knn_io


def bi_retrieve_candidates(q, index, candidate_region):
    IO = 0
    vd = index.voronoi_diagram
    points = index.points
    candidates = list()
    (nn_o, nn_p, nn_dist), nn_io = index.nearest(q)
    IO += nn_io
    visited = {nn_o}
    buffer = [(nn_o, nn_p)]
    while len(buffer) > 0:
        o, p = buffer.pop()
        if candidate_region.intersects(p):
            candidates.append((o, p))
        for neighbor in vd.neighbors(o):
            if neighbor not in visited:
                visited.add(neighbor)
                if candidate_region.intersects(vd.cell(neighbor)):
                    neighbor_p = points[neighbor]
                    IO += 1
                    buffer.append((neighbor, neighbor_p))
    return candidates, IO


def kNN(index, q, k, cache, nn=None):
    IO = 0
    knn = list()
    h = MinHeap()
    if nn is None:
        (nn_o, nn_p, nn_dist), nn_io = index.NN(q)
        cache[nn_o] = nn_p
        IO += nn_io
    else:
        nn_o, nn_p, nn_dist = nn
    vd = index.voronoi_diagram
    points = index.points
    h.push((nn_dist, nn_o, nn_p))
    visited = {nn_o}
    count = 0
    while count < k and len(h) > 0:
        dist, o, p = h.pop()
        knn.append((o, p, dist))
        for neighbor in vd.neighbors(o):
            if neighbor not in visited:
                if neighbor in cache:
                    neighbor_p = cache[neighbor]
                else:
                    neighbor_p = points[neighbor]
                    cache[neighbor] = neighbor_p
                    IO += 1
                visited.add(neighbor)
                h.push((points[neighbor].distance(q), neighbor, neighbor_p))
        count += 1
    return knn, IO
