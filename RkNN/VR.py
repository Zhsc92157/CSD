from time import time

import common


def MonoRkNN(q_id, k, index, with_statistics=False):
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
    if with_statistics:
        return result, pruning_time - begin_time, verification_time - pruning_time, pruning_io, verification_io, candidate_num, verified_candidate_num
    else:
        return result


def mono_pruning(q_id, q, k, index, cache):
    IO = 0
    partitions = list(common.partitions(q, index.space, 6))
    region_list = list()
    points = index.points
    vd = index.voronoi_diagram
    visited = {q_id}
    H = common.MinHeap()
    S = [common.MaxHeap() for i in range(6)]
    for neighbor_id in vd.neighbors(q_id):
        if neighbor_id in cache:
            neighbor_p = cache[neighbor_id]
        else:
            neighbor_p = points[neighbor_id]
            cache[neighbor_id] = neighbor_p
            IO += 1
        H.push((1, neighbor_id, neighbor_p))
        visited.add(neighbor_id)
    while len(H) > 0:
        gd_p, p_id, p = H.pop()
        for i in range(6):
            if partitions[i].intersects(p):
                if len(S[i]) < k:
                    dist_bound = float('inf')
                else:
                    dist_bound = S[i].first()[0]
                dist_p = p.distance(q)
                if gd_p <= k and dist_p < dist_bound:
                    S[i].push((dist_p, p_id, p))
                    for neighbor_id in vd.neighbors(p_id):
                        if neighbor_id not in visited:
                            if neighbor_id in cache:
                                neighbor_p = cache[neighbor_id]
                            else:
                                neighbor_p = points[neighbor_id]
                                cache[neighbor_id] = neighbor_p
                                IO += 1
                            gd_neighbor = gd_p + 1
                            visited.add(neighbor_id)
                            H.push((gd_neighbor, neighbor_id, neighbor_p))
    candidate_idx = set()
    candidates = list()
    for i in range(6):
        s = sorted(S[i])
        if len(s) >= k:
            s = s[:k]
        for dist, p_id, p in s:
            if p_id not in candidate_idx:
                candidate_idx.add(p_id)
                candidates.append((p_id, p, dist))
    return candidates, IO


def mono_knn_radius(q_id, q, k, index, cache):
    knn, knn_io = kNN(index, q, k + 1, cache, (q_id, q, 0))
    return knn[-1][2], knn_io


def mono_verification(q_id, q, k, candidates, index, cache):
    IO = 0
    rknn = list()
    candidates
    for c_o, c_p, c_dist in candidates:
        r, knn_io = mono_knn_radius(c_o, c_p, k, index, cache)
        IO += knn_io
        if c_dist <= r:
            rknn.append((c_o, c_p))
    return rknn, IO, len(candidates), len(candidates)


def BiRkNN(q_id, k, facility_index, user_index, with_statistics=False):
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
    if with_statistics:
        return result, pruning_time - begin_time, verification_time - pruning_time, pruning_io, verification_io, candidate_num, verified_candidate_num
    else:
        return result


def bi_pruning(q_id, q, k, index, cache):
    IO = 0
    partitions = list(common.partitions(q, index.space, 6))
    region_list = list()
    points = index.points
    vd = index.voronoi_diagram
    visited = {q_id}
    H = common.MinHeap()
    S = [common.MaxHeap() for i in range(6)]
    for neighbor_id in vd.neighbors(q_id):
        if neighbor_id in cache:
            neighbor_p = cache[neighbor_id]
        else:
            neighbor_p = points[neighbor_id]
            cache[neighbor_id] = neighbor_p
            IO += 1
        H.push((1, neighbor_id, neighbor_p))
        visited.add(neighbor_id)
    while len(H) > 0:
        gd_p, p_id, p = H.pop()
        for i in range(6):
            if partitions[i].intersects(p):
                if len(S[i]) < k:
                    dist_bound = float('inf')
                else:
                    dist_bound = S[i].first()[0]
                dist_p = p.distance(q)
                if gd_p <= k and dist_p < dist_bound:
                    S[i].push((dist_p, p_id, p))
                    for neighbor_id in vd.neighbors(p_id):
                        if neighbor_id not in visited:
                            if neighbor_id in cache:
                                neighbor_p = cache[neighbor_id]
                            else:
                                neighbor_p = points[neighbor_id]
                                cache[neighbor_id] = neighbor_p
                                IO += 1
                            gd_neighbor = gd_p + 1
                            visited.add(neighbor_id)
                            H.push((gd_neighbor, neighbor_id, neighbor_p))
    for i in range(6):
        s = sorted(S[i])
        if len(s) >= k:
            r = s[k - 1][0]
        elif len(s) > 0:
            r = s[-1][0]
        else:
            r = 0
        if r > 0:
            region_list.append(common.sector(q, r, partitions[i].angles).buffer(0.01))
    region = reduce(lambda x, y: x.union(y), region_list)
    return region, IO


def bi_verification(q, k, candidate_region, facility_index, user_index, facility_cache):
    IO = 0
    rknn = list()
    candidates, retrieve_candidates_io = bi_retrieve_candidates(user_index, candidate_region)
    for c in candidates:
        c_o, c_p = c
        r, knn_io = bi_knn_radius(c_p, k, facility_index, facility_cache)
        IO += knn_io
        if c_p.distance(q) <= r:
            rknn.append(c)
    return rknn, IO, len(candidates), len(candidates)


def bi_knn_radius(q, k, index, cache):
    knn, knn_io = kNN(index, q, k, cache)
    return knn[-1][2], knn_io


def NN(index, q):
    IO = 0
    h = common.MinHeap()
    h.push((0, index.root))
    IO += 1
    best_dist = float('inf')
    nn = None
    while len(h) > 0:
        e_dist, e = h.pop()
        if e.is_leaf_node:
            for child in e.children:
                IO += 1
                c_dist = child.geom.distance(q)
                if c_dist < best_dist:
                    nn = child
                    best_dist = c_dist
            if index.voronoi_diagram.cell(nn.obj).intersects(q):
                return (nn.obj, nn.geom, best_dist), IO
        else:
            for child in e.children:
                IO += 1
                c_dist = child.geom.distance(q)
                h.push((c_dist, child))
    return (nn.obj, nn.geom, best_dist), IO


def kNN(index, q, k, cache, nn=None):
    IO = 0
    knn = list()
    h = common.MinHeap()
    if nn is None:
        (nn_o, nn_p, nn_dist), nn_io = NN(index, q)
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


def bi_retrieve_candidates(index, candidate_region):
    candidates = list()
    IO = 0
    entries = {index.root}
    IO += 1
    while len(entries) > 0:
        e = entries.pop()
        if candidate_region.intersects(e.geom):
            if e.is_data_node:
                candidates.append((e.obj, e.geom))
            else:
                for child in e.children:
                    entries.add(child)
                    IO += 1
    return candidates, IO
