from collections import deque
from typing import Callable

import numpy as np

from . import cvrp, get_logger


logger = get_logger(__name__)

# constructor of initial feasible solution
SolutionConstructor = Callable[[cvrp.Problem], cvrp.Solution]


def random_builder(p: cvrp.Problem) -> cvrp.Solution:
    visited = {0}
    unvisited = set(range(p.dim)) - visited
    routes, demands = [], []

    while unvisited:
        c, *_ = unvisited
        route = [c]
        demand = p.demands[c]

        while unvisited:
            c, *_ = unvisited
            c_demand = p.demands[c]

            if c_demand + demand > p.capacity:
                break

            route.append(c)
            unvisited.remove(c)
            demand += c_demand

        routes.append(route)
        demands.append(demand)

    return cvrp.Solution(
        routes=routes,
        cost=p.solution_cost(routes),
    )


def neighbours(customer: int, d: cvrp.DistanceMatrix) -> np.ndarray:
    locations = np.argsort(d[customer])
    return locations[locations != 0]


def nearest_neighbour_builder(p: cvrp.Problem) -> cvrp.Solution:
    seeds = deque(fps_seed(p.distances, p.min_vehicles))
    visited = set(seeds) | {0}
    unvisited = set(range(p.dim)) - visited
    routes, demands = [], []

    while unvisited:
        seeds.append(0)
        seed = seeds.popleft()
        route = [seed]
        demand = p.demands[seed]

        while unvisited:
            head = route[-1]
            # TODO n.teterin: support insertions with regret
            nearest_customer, *_ = [
                n for n in neighbours(head, d=p.distances) if n in unvisited
            ]
            customer_demand = p.demands[nearest_customer]
            if customer_demand + demand > p.capacity:
                break

            route.append(nearest_customer)
            unvisited.remove(nearest_customer)
            demand += customer_demand

        routes.append(route)
        demands.append(demand)

    for route in routes:
        # remove the depot
        if 0 in route:
            route.remove(0)

    # for customer, _ in enumerate(p.customers):
    #     if customer == 0:
    #         continue

    #     if not any(customer in r for r in routes):
    #         logger.error(f"{customer} not inserted after repair!")
    #         raise RuntimeError("boo!")

    return cvrp.Solution(
        routes=routes,
        cost=p.solution_cost(routes),
    )


def fps_seed(
    distances: cvrp.DistanceMatrix,
    n_vehicles: int,
) -> cvrp.Route:
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
    return seeds[1:]
