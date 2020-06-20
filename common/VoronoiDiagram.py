from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import random

from shapely.geometry import MultiPoint, Point
from shapely.prepared import prep


# from common.geometric_util import mid_point, distance, norm


class VoronoiDiagram(Voronoi):
    def __init__(self, data, lower_bounds, upper_bounds):
        self.diagonal = Point(lower_bounds).distance(Point(upper_bounds))
        points = [e[1] for e in data]
        objects = [e[0] for e in data]
        Voronoi.__init__(self, [(p.x, p.y) for p in points])
        self.neighbor_dict = {o: set() for o in objects}
        self.cell_dict = {o: set() for o in objects}
        center = self.points.mean(axis=0)
        for pointidx, simplex in zip(self.ridge_points, self.ridge_vertices):
            o0 = objects[pointidx[0]]
            o1 = objects[pointidx[1]]
            self.neighbor_dict[o0].add(o1)
            self.neighbor_dict[o1].add(o0)
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                finite_vertex = self.vertices[simplex[simplex >= 0][0]]  # finite end Voronoi vertex
                t = self.points[pointidx[1]] - self.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = self.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = finite_vertex + direction * self.diagonal
                ridge_vertices = (tuple(finite_vertex), tuple(far_point))
                self.cell_dict[o0].add(ridge_vertices[0])
                self.cell_dict[o1].add(ridge_vertices[0])
                self.cell_dict[o0].add(ridge_vertices[1])
                self.cell_dict[o1].add(ridge_vertices[1])
            else:
                ridge_vertices = tuple(tuple(v) for v in self.vertices[simplex])
                self.cell_dict[o0].add(ridge_vertices[0])
                self.cell_dict[o1].add(ridge_vertices[0])
                self.cell_dict[o0].add(ridge_vertices[1])
                self.cell_dict[o1].add(ridge_vertices[1])
        for o in objects:
            cell_points = self.cell_dict[o]
            self.cell_dict[o] = MultiPoint(list(cell_points)).convex_hull

    def plot(self, ax):
        if self.points.shape[1] != 2:
            raise ValueError("Voronoi diagram is not 2-D")
        ptp_bound = self.points.ptp(axis=0)
        border_width = 0.01
        ax.set_xlim(self.points[:, 0].min() - border_width * ptp_bound[0],
                    self.points[:, 0].max() + border_width * ptp_bound[0])
        ax.set_ylim(self.points[:, 1].min() - border_width * ptp_bound[1],
                    self.points[:, 1].max() + border_width * ptp_bound[1])
        ax.plot(self.points[:, 0], self.points[:, 1], '.')
        # ax.plot(self.vertices[:, 0], self.vertices[:, 1], '.')
        for cell in self.cell_dict.values():
            points = list(cell.exterior.coords)
            ax.plot([p[0] for p in points], [p[1] for p in points], '-', color='lightgray')

    @property
    def point_indices(self):
        return self.neighbor_dict.keys()

    def neighbors(self, i):
        return self.neighbor_dict[i]

    def size(self):
        return len(self.points)

    def cell(self, i):
        return self.cell_dict[i]

# if __name__ == '__main__':
#     n = 100
#     bounds = [0,1000]
#     points = generate_points(bounds, n)
#     vd = VoronoiDiagram(points, (bounds[0],bounds[0]),(bounds[1],bounds[1]))
#     print vd.cell_dict
#     ax = plt.gca()
#     vd.plot(ax)
#     plt.show()
