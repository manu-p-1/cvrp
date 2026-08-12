"""
https://github.com/manu-p-1/cvrp
distance.py

Precomputed all-pairs Euclidean distance matrix for O(1) pairwise distance
lookups. Eliminates repeated square-root computations during fitness
evaluation and local search, which is the single largest performance
bottleneck in evolutionary CVRP solvers.

Reference:
    Toth, P. and Vigo, D. (2002). "The Vehicle Routing Problem."
    SIAM Monographs on Discrete Mathematics and Applications, Chapter 1.
"""

import math


class DistanceMatrix:
    """All-pairs distance matrix indexed by node ID.

    Distances are rounded to the nearest integer (Euclidean CVRP convention).
    Storage is a flat 2-D list for cache-friendly O(1) access.
    """

    __slots__ = ('_d', '_size')

    def __init__(self, depot, buildings):
        """Build the distance matrix from depot + customer nodes.

        :param depot: The depot Building object
        :param buildings: Iterable of customer Building objects
        """
        all_nodes = [depot] + list(buildings)
        max_id = max(n.node for n in all_nodes)
        size = max_id + 1

        d = [[0] * size for _ in range(size)]
        for i, b1 in enumerate(all_nodes):
            for b2 in all_nodes[i + 1:]:
                val = round(math.sqrt((b1.x - b2.x) ** 2 + (b1.y - b2.y) ** 2))
                d[b1.node][b2.node] = val
                d[b2.node][b1.node] = val

        self._d = d
        self._size = size

    def dist(self, b1, b2):
        """Distance between two Building objects (O(1) lookup)."""
        return self._d[b1.node][b2.node]

    def dist_by_id(self, n1, n2):
        """Distance between two node IDs (O(1) lookup)."""
        return self._d[n1][n2]

    @property
    def size(self):
        """Number of rows/columns in the matrix."""
        return self._size
