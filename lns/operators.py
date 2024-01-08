from dataclasses import dataclass, field
from functools import partial
from itertools import chain, pairwise
from typing import Callable, Optional, Tuple

import numpy as np

from . import cvrp

DestroyOperator = Callable[[cvrp.Solution], cvrp.PartialSolution]
RepairOperator = Callable[[cvrp.PartialSolution], Optional[cvrp.Solution]]
# https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/


def remove_unassigned(sol: cvrp.Solution, unassigned: cvrp.Unassigned):
    # helper function that creates partial solution from
    # a given complete solution and set of customers to drop
    return cvrp.PartialSolution(
        routes=[[c for c in route if c not in unassigned] for route in sol.routes],
        unassigned=unassigned,
    )


@dataclass(frozen=True)
class BasicDestroyConfig:
    dim: int
    bounds: Tuple[int, int]
    rng: np.random.Generator = field(
        default_factory=partial(np.random.default_rng, seed=42)
    )

    def nodes_to_remove(self) -> int:
        a, b = self.bounds
        return self.rng.integers(a, b)


@dataclass(frozen=True)
class RandomRemove:
    # removes randomly sampled points from solution
    cfg: BasicDestroyConfig

    def __call__(self, sol: cvrp.Solution):
        n = self.cfg.nodes_to_remove()
        dim = self.cfg.dim
        removed_customers = set(self.rng.choice(dim, size=n, replace=False))
        return remove_unassigned(sol, removed_customers)


@dataclass(frozen=True)
class WorstRemove:
    # removes points with highest cost (hoping that repair
    # operators will find a better place to emplace them)
    route_cost_hook: Callable[[cvrp.Route], float]
    cfg: BasicDestroyConfig

    def __call__(self, sol: cvrp.Solution):
        dim = self.cfg.dim
        savings = np.empty(dim, dtype=float)

        for route in sol.routes:
            route_cost = self.route_cost_hook(route)
            route_copy = np.asarray(route)
            for customer in route:
                r = route_copy[route_copy != customer]
                savings[customer] = route_cost - self.route_cost_hook(r)

        n = self.cfg.nodes_to_remove()
        worst_nodes = -savings[1:].argsort()[:n]
        return remove_unassigned(sol, set(worst_nodes))


@dataclass(frozen=True)
class BasicRepairConfig:
    problem: cvrp.Problem
    rng: np.random.Generator = field(
        default_factory=partial(np.random.default_rng, seed=42)
    )


def insert_costs(customer: int, route: cvrp.Route, d: cvrp.DistanceMatrix):
    # computes insertion costs for given route and customer
    costs = np.empty(len(route) + 1, dtype=float)
    tour = chain([0], route, [0])
    for i, (a, b) in enumerate(pairwise(tour)):
        costs[i] = d[a, customer] + d[customer, b] - d[a, b]

    return costs


@dataclass(frozen=True)
class GreedyRepair:
    cfg: BasicRepairConfig

    def __call__(self, partial_sol: cvrp.PartialSolution) -> cvrp.Solution:
        # for each unassigned route, find best
        # position to insert
        unassigned = np.fromiter(
            partial_sol.unassigned, count=len(partial_sol.unassigned)
        )
        self.cfg.rng.shuffle(unassigned)
        routes = partial_sol.routes

        for customer in unassigned:
            route, idx = self.best_insert(customer, routes)
            if route is None:
                # capacity constraints can't be satisfied
                # with current number of vehicles,
                # create a new route with single customer
                routes.append([customer])
                continue

            route.insert(idx, customer)

        return cvrp.Solution(
            routes=routes,
            cost=self.cfg.problem.solution_cost(routes),
        )

    def best_insert(
        self,
        customer: int,
        routes: cvrp.Routes,
    ) -> Tuple[Optional[cvrp.Route], Optional[int]]:
        # searches for the best position to insert
        route_loads = tuple(self.cfg.problem.route_cost(r) for r in routes)
        demand = self.cfg.problem.demands[customer]
        capacity = self.cfg.problem.capacity
        distances = self.cfg.problem.distances

        best_cost, best_route, best_idx = np.inf, None, None

        for route, load in zip(routes, route_loads):
            if demand + load > capacity:
                continue

            costs = insert_costs(customer, route, distances)
            # idx is in range [0, len(route)], hence is valid
            # for insertion (as Python lists would insert before given index,
            # e.g. inserting at 0 means before the first element)
            idx = costs.argmin()
            cost = costs[idx]
            if cost < best_cost:
                best_cost = cost
                best_route = route
                best_idx = idx

        return best_route, best_idx


# TODO n.teterin:
# implement 2-regret greedy repair operator
# and K-regret repair operator
