import os
import random
import numpy as np

from shapely.geometry import Point

# from RkNN import *
from RkNN import VR_BiRkNN, CSD_MonoRkNN, SLICE_BiRkNN, CSD_BiRkNN, VR_MonoRkNN, SLICE_MonoRkNN
from common.Rtree import RtreeIndex
from common.VoKDtree import VoKDtreeIndex
from common.VoRtee import VoRtreeIndex
import matplotlib.pyplot as plt


def generate_points(n, mode='Uniform', bounds=None):
    if mode == 'Uniform':
        if bounds is None:
            bounds = (0, 0, 10000, 10000)
        p_set = set()
        data = []
        while len(p_set) < n:
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            if (x, y) not in p_set:
                p_set.add((x, y))
                data.append((len(p_set) - 1, Point(x, y)))
        return data

    if mode == 'Normal':
        if bounds is None:
            bounds = (0, 0, 10000, 10000)
        p_set = set()
        x_mu = (bounds[0] + bounds[2]) / 2
        x_sigma = abs(bounds[0] - bounds[2]) / 10
        y_mu = (bounds[1] + bounds[3]) / 2
        y_sigma = abs(bounds[1] - bounds[3]) / 10
        data = []
        while len(p_set) < n:
            x = np.random.normal(x_mu, x_sigma)
            y = np.random.normal(y_mu, y_sigma)
            if (x, y) not in p_set and bounds[0] < x < bounds[2] and bounds[1] < y < bounds[3]:
                p_set.add((x, y))
                data.append((len(p_set) - 1, Point(x, y)))
        return data

    if mode == 'Real':
        if bounds is None:
            bounds = (-125.733611, 23.546944, -65.95, 50.361794)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/us50000.txt')
        f = open(path)
        point_pool = list()
        visited = set()
        for line in f.readlines():
            line_str = line.strip().split()
            x = float(line_str[1])
            y = float(line_str[0])
            if (x, y) not in visited:
                visited.add((x, y))
                point_pool.append((x, y))
        points = random.sample(point_pool, n)
        data = [(i, Point(points[i][0], points[i][1])) for i in range(len(points))]
        return data


class MonoTest:
    def __init__(self, num, distribution='Uniform'):
        self.num = num
        self.distribution = distribution
        if distribution == 'Real':
            self.bounds = (-125.733611, 23.546944, -65.95, 50.361794)
        else:
            self.bounds = (0, 0, 10000, 10000)
        self.points = generate_points(self.num, self.distribution)

    def run(self, k, times):
        print
        print '{}(k={}, number={}, distribution={}) '.format(
            self.algorithm.name, k, self.num, self.distribution)
        total_pruning_cost = 0
        total_verification_cost = 0
        total_pruning_io = 0
        total_verification_io = 0
        total_candidate_num = 0
        total_verified_candidate_num = 0
        width = 152

        print '-' * width
        print '| Instance | Pruning time (ms) | Pruning IO | Verification time (ms) | Verification IO | Total time (' \
              'ms) | Total IO | #candidate | #verified candidate |'
        print '-' * width
        for i in range(times):
            q_id = random.randint(0, self.num - 1)
            result, pruning_cost, verification_cost, pruning_io, verification_io, candidate_num, verified_candidate_num = self.algorithm(
                q_id, k, self.index)
            total_pruning_cost += pruning_cost
            total_verification_cost += verification_cost
            total_pruning_io += pruning_io
            total_verification_io += verification_io
            total_candidate_num += candidate_num
            total_verified_candidate_num += verified_candidate_num
            print '| {:^8} | {:>17.2f} | {:>10} | {:>22.2f} | {:>15} | {:>15.2f} | {:>8} | {:>10} | {:>19} |'.format(
                i + 1,
                pruning_cost * 1000, pruning_io,
                verification_cost * 1000, verification_io,
                pruning_cost * 1000 + verification_cost * 1000, pruning_io + verification_io, candidate_num,
                verified_candidate_num)
        print '-' * width
        print '|  Average | {:>17.2f} | {:>10.2f} | {:>22.2f} | {:>15.2f} | {:>15.2f} | {:>8.2f} | {:>10.2f} | {:>19.2f} |'.format(
            total_pruning_cost / times * 1000, total_pruning_io / float(times),
            total_verification_cost / times * 1000, total_verification_io / float(times),
            total_pruning_cost / times * 1000 + total_verification_cost / times * 1000,
            total_pruning_io / float(times) + total_verification_io / float(times), total_candidate_num / float(times),
            total_verified_candidate_num / float(times))
        print '-' * width
        print


class CSDMonoTest(MonoTest):
    def __init__(self, num, distribution='Uniform'):
        MonoTest.__init__(self, num, distribution)
        self.index = VoKDtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                   self.bounds[3],
                                   self.points)
        self.algorithm = CSD_MonoRkNN


class SLICEMonoTest(MonoTest):
    def __init__(self, num, distribution='Uniform'):
        MonoTest.__init__(self, num, distribution)
        self.index = RtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                self.bounds[3],
                                self.points)
        self.algorithm = SLICE_MonoRkNN


class VRMonoTest(MonoTest):
    def __init__(self, num, distribution='Uniform'):
        MonoTest.__init__(self, num, distribution)
        self.index = VoRtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                  self.bounds[3],
                                  self.points)
        self.algorithm = VR_MonoRkNN


class BiTest:
    def __init__(self, user_num, facility_num, user_distribution='Uniform', facility_distribution='Uniform'):
        self.user_num = user_num
        self.facility_num = facility_num
        self.user_type = user_distribution
        self.facility_distribution = facility_distribution
        if user_distribution == 'Real' or facility_distribution == 'Real':
            self.bounds = (-125.733611, 23.546944, -65.95, 50.361794)
        else:
            self.bounds = (0, 0, 10000, 10000)
        self.facility = generate_points(self.facility_num, self.facility_distribution,
                                        self.bounds)
        self.user = generate_points(self.user_num, user_distribution, self.bounds)

    def plot(self, k):
        q_id = random.randint(0, self.facility_num - 1)
        result, pruning_cost, verification_cost, pruning_io, verification_io, candidate_num, verified_candidate_num = self.algorithm.BiRkNN(
            q_id, k,
            self.facility_index,
            self.user_index)
        q = self.facility_index.points[q_id]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_xlim(self.facility_bounds[0], self.facility_bounds[2])
        ax.set_ylim(self.facility_bounds[1], self.facility_bounds[3])
        for o, p in result:
            plt.plot(p.x, p.y, '.', color='blue', markersize=4)
        plt.plot(q.x, q.y, '.', color='red', markersize=15)
        plt.show()

    def run(self, k, times):
        print
        print '{}(k={}) User(number={}, distribution={}) Facility(number={}, distribution={})'.format(
            self.algorithm.name, k, self.user_num, self.user_type, self.facility_num, self.facility_distribution)
        total_pruning_cost = 0
        total_verification_cost = 0
        total_pruning_io = 0
        total_verification_io = 0
        total_candidate_num = 0
        total_verified_candidate_num = 0
        width = 152

        print '-' * width
        print '| Instance | Pruning time (ms) | Pruning IO | Verification time (ms) | Verification IO | Total time (' \
              'ms) | Total IO | #candidate | #verified candidate |'
        print '-' * width
        for i in range(times):
            q_id = random.randint(0, self.facility_num - 1)
            result, pruning_cost, verification_cost, pruning_io, verification_io, candidate_num, verified_candidate_num = self.algorithm(
                q_id, k,
                self.facility_index,
                self.user_index)
            total_pruning_cost += pruning_cost
            total_verification_cost += verification_cost
            total_pruning_io += pruning_io
            total_verification_io += verification_io
            total_candidate_num += candidate_num
            total_verified_candidate_num += verified_candidate_num
            print '| {:^8} | {:>17.2f} | {:>10} | {:>22.2f} | {:>15} | {:>15.2f} | {:>8} | {:>10} | {:>19} |'.format(
                i + 1,
                pruning_cost * 1000, pruning_io,
                verification_cost * 1000, verification_io,
                pruning_cost * 1000 + verification_cost * 1000, pruning_io + verification_io, candidate_num,
                verified_candidate_num)
        print '-' * width
        print '|  Average | {:>17.2f} | {:>10.2f} | {:>22.2f} | {:>15.2f} | {:>15.2f} | {:>8.2f} | {:>10.2f} | {:>19.2f} |'.format(
            total_pruning_cost / times * 1000, total_pruning_io / float(times),
            total_verification_cost / times * 1000, total_verification_io / float(times),
            total_pruning_cost / times * 1000 + total_verification_cost / times * 1000,
            total_pruning_io / float(times) + total_verification_io / float(times), total_candidate_num / float(times),
            total_verified_candidate_num / float(times))
        print '-' * width
        print


class VRBiTest(BiTest):
    def __init__(self, user_num, facility_num, user_distribution='Uniform', facility_type='Uniform'):
        BiTest.__init__(self, user_num, facility_num, user_distribution, facility_type)
        self.user_index = VoRtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                       self.bounds[3],
                                       self.user)
        self.facility_index = VoRtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                           self.bounds[3], self.facility)
        self.algorithm = VR_BiRkNN


class SLICEBiTest(BiTest):
    def __init__(self, user_num, facility_num, user_distribution='Uniform', facility_type='Uniform'):
        BiTest.__init__(self, user_num, facility_num, user_distribution, facility_type)
        self.user_index = RtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3],
                                     self.user)
        self.facility_index = RtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                         self.bounds[3], self.facility)
        self.algorithm = SLICE_BiRkNN


class CSDBiTest(BiTest):
    def __init__(self, user_num, facility_num, user_distribution='Uniform', facility_type='Uniform'):
        BiTest.__init__(self, user_num, facility_num, user_distribution, facility_type)
        self.user_index = VoKDtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                        self.bounds[3],
                                        self.user)
        self.facility_index = VoKDtreeIndex(self.bounds[0], self.bounds[1], self.bounds[2],
                                            self.bounds[3], self.facility)
        self.algorithm = CSD_BiRkNN


def Effect_of_k_on_MonoRkNN():
    time = 10
    # uniform
    num = 100000
    test = CSDMonoTest(num, 'Uniform')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)

    test = SLICEMonoTest(num, 'Uniform')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)

    test = VRMonoTest(num, 'Uniform')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)

    # real
    num = 49601
    test = CSDMonoTest(num, 'Real')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)

    test = SLICEMonoTest(num, 'Real')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)

    test = VRMonoTest(num, 'Real')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)


def Effect_of_k_on_BiRkNN():
    time = 10
    # uniform
    user_num = 100000
    facility_num = 100000
    test = CSDBiTest(user_num, facility_num, 'Uniform', 'Uniform')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)
    test = SLICEBiTest(user_num, facility_num, 'Uniform', 'Uniform')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)
    test = VRBiTest(user_num, facility_num, 'Uniform', 'Uniform')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)
    # real
    user_num = 49601
    facility_num = 49601
    test = CSDBiTest(user_num, facility_num, 'Real', 'Real')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)
    test = SLICEBiTest(user_num, facility_num, 'Real', 'Real')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)
    test = VRBiTest(user_num, facility_num, 'Real', 'Real')
    for i in range(0, 4):
        k = 10 ** i
        test.run(k, time)


def Effect_of_data_size_on_MonoRkNN():
    time = 10
    k = 100
    for i in range(3, 7):
        test = SLICEMonoTest(10 ** i)
        test.run(k, time)
    for i in range(3, 7):
        test = SLICEMonoTest(10 ** i)
        test.run(k, time)
    for i in range(3, 7):
        test = VRMonoTest(10 ** i)
        test.run(k, time)


def Effect_of_data_size_on_BiRkNN():
    time = 10
    k = 100
    for i in range(3, 7):
        test = CSDBiTest(10 ** i, 10 ** i)
        test.run(k, time)
    for i in range(3, 7):
        test = SLICEBiTest(10 ** i, 10 ** i)
        test.run(k, time)
    for i in range(3, 7):
        test = VRBiTest(10 ** i, 10 ** i)
        test.run(k, time)


def Effect_of_user_per_facility():
    print 'bi_effect_user_per_facility'
    time = 10
    k = 100
    facility_num = 100000
    for i in range(0, 4):
        user_num = int(facility_num * 0.5 * (2 ** i))
        test = CSDBiTest(user_num, facility_num)
        test.run(k, time)
    for i in range(0, 4):
        user_num = int(facility_num * 0.5 * (2 ** i))
        test = SLICEBiTest(user_num, facility_num)
        test.run(k, time)
    for i in range(0, 4):
        user_num = int(facility_num * 0.5 * (2 ** i))
        test = VRBiTest(user_num, facility_num)
        test.run(k, time)


def Effect_of_distribution():
    time = 10
    k = 100
    distributions = [('Uniform', 'Uniform'), ('Uniform', 'Normal'), ('Uniform', 'Real'), ('Normal', 'Uniform'),
                     ('Normal', 'Normal'), ('Normal', 'Real'), ('Real', 'Uniform'), ('Real', 'Normal'),
                     ('Real', 'Real')]
    num = 49601
    for d in distributions:
        test = CSDBiTest(num, num, d[0], d[1])
        test.run(k, time)
    for d in distributions:
        test = SLICEBiTest(num, num, d[0], d[1])
        test.run(k, time)
    for d in distributions:
        test = VRBiTest(num, num, d[0], d[1])
        test.run(k, time)


if __name__ == '__main__':
    Effect_of_data_size_on_MonoRkNN()
    Effect_of_data_size_on_BiRkNN()
    Effect_of_k_on_MonoRkNN()
    Effect_of_k_on_BiRkNN()
    Effect_of_distribution()
    Effect_of_user_per_facility()
