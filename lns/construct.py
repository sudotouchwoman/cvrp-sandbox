from typing import Callable
import numpy as np
from . import cvrp


# constructor of initial feasible solution
SolutionConstructor = Callable[[cvrp.Problem], cvrp.Solution]


def neighbours(customer: int, d: cvrp.DistanceMatrix) -> np.ndarray:
    locations = np.argsort(d[customer])
    return locations[locations != 0]


def nearest_neighbour_builder(p: cvrp.Problem) -> cvrp.Solution:
    seeds = fps_seed(p.distances, p.max_vehicles)
    visited = set(seeds)
    # note the preceeding 0
    routes = [[0, s] for s in seeds]
    demands = [p.demands[s] for s in seeds]
    unvisited = set(x for x in range(1, p.dim) if x not in visited)

    while unvisited:
        # this is not exactly true and can possibly lead
        # to a situation when all routes are fully loaded, but on average,
        # distance based seeding should work at least better than random
        for i, (demand, route) in enumerate(zip(demands, routes)):
            head = route[-1]
            # TODO n.teterin: support insertions with regret
            nearest_customer, *_ = [n for n in neighbours(head) if n in unvisited]
            customer_demand = p.demands[nearest_customer]
            if customer_demand + demand > p.capacity:
                break

            route.append(nearest_customer)
            unvisited.remove(nearest_customer)
            demands[i] += customer_demand

    for i, route in enumerate(routes):
        # remove the depot
        routes[i] = route[1:]

    return cvrp.Solution(
        routes=routes,
        cost=p.solution_cost(routes),
    )


def fps_seed(
    distances: cvrp.DistanceMatrix,
    n_vehicles: int,
) -> cvrp.Routes:
    # seeds routes for NN builder via
    # farthest point sampling

    # adapted from
    # https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
    seeds = np.zeros(n_vehicles + 1, dtype=np.int64)
    ds = np.asarray(distances[0, :])

    for i in range(1, n_vehicles + 1):
        point = ds.argmax()
        seeds[i] = point
        ds = np.minimum(ds, distances[point, :])

    # omit the first point, which
    # represents the depot
    return [[s] for s in seeds[1:]]
