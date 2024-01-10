from dataclasses import dataclass
from itertools import pairwise
from typing import Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from . import get_logger


logger = get_logger(__name__)

DistanceMatrix = Mapping[Tuple[int, int], float]
Route = Sequence[int]
Routes = Sequence[Route]
Unassigned = Set[int]
Demands = Sequence[float]


@dataclass
class PartialSolution:
    routes: Routes
    unassigned: Unassigned


@dataclass
class Solution:
    routes: Routes
    cost: float

    def find_route(self, customer: int) -> Route:
        for r in self.routes:
            if customer in r:
                return r

        raise ValueError(f"customer {customer} missing in solution")

    @classmethod
    def from_vrplib(cls, sol):
        return cls(**sol)


@dataclass
class Problem:
    customers: Sequence[Tuple[float, float]]
    distances: DistanceMatrix
    demands: Demands
    capacity: float
    min_vehicles: int
    name: str = ""

    @property
    def dim(self) -> int:
        return len(self.distances)

    @property
    def depot(self):
        return self.customers[0]

    def solution_cost(self, routes: Routes):
        return solution_cost(routes, self.distances)

    def route_cost(self, route: Route):
        return route_cost(route, self.distances)

    def solution_load(self, routes: Routes):
        return sum(self.route_load(r) for r in routes)

    def route_load(self, route: Route):
        return sum(self.demands[c] for c in route)

    @classmethod
    def from_vrplib(cls, data):
        # TODO n.teterin: support problems without
        # explicit estimates on truck count
        _, _, k = data["name"].split("-")
        k = int(k.removeprefix("k"))
        return cls(
            customers=data.get("node_coord", []),
            distances=data.get("edge_weight", np.empty(0)),
            demands=data.get("demand", []),
            capacity=data.get("capacity", np.inf),
            min_vehicles=k,
            name=data["name"] + " " + data["comment"],
        )


def route_cost(route: Route, distance_matrix: DistanceMatrix):
    i, j = route[0], route[-1]
    inner_cost = sum(distance_matrix[x] for x in pairwise(route))
    return distance_matrix[0, i] + distance_matrix[j, 0] + inner_cost


def solution_cost(routes: Routes, distance_matrix: DistanceMatrix):
    return sum(route_cost(r, distance_matrix) for r in routes)


def check_solution(p: Problem, solution: Solution) -> bool:
    is_ok = True

    for i, route in enumerate(solution.routes):
        demand = sum(p.demands[i] for i in route)
        if demand > p.capacity:
            logger.warning(
                f"route {i} demand exceeds capacity: {demand} > {p.capacity}"
            )
            is_ok = False

    return is_ok


# partial solution is what is obtained by destroy operators
# solution is what is produced by repair operators and
# (possibly) Local Search methods

# problem holds the metadata for heuristics to operate:
# list of all customer features, precomputed distance matrix,
# and the best solution so far along with current solution (which
# can of course be worse)
