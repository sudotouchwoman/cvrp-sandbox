from dataclasses import dataclass, field
from functools import partial
from itertools import chain, pairwise
from typing import Callable, Optional, Tuple

import numpy as np

from lns import construct

from . import cvrp, get_logger


logger = get_logger(__name__)

DestroyOperator = Callable[[cvrp.Solution], cvrp.PartialSolution]
RepairOperator = Callable[[cvrp.PartialSolution], Optional[cvrp.Solution]]
# https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/


def remove_unassigned(sol: cvrp.Solution, unassigned: cvrp.Unassigned):
    # helper function that creates partial solution from
    # a given complete solution and set of customers to drop
    routes = ([c for c in route if c not in unassigned] for route in sol.routes)
    # remember to remove empty routes, if any
    return cvrp.PartialSolution(
        routes=[r for r in routes if r],
        unassigned=unassigned,
    )


@dataclass(frozen=True)
class BasicDestroyConfig:
    problem: cvrp.Problem
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
        dim = self.cfg.problem.dim
        removed_customers = set(self.cfg.rng.choice(dim, size=n, replace=False))
        return remove_unassigned(sol, removed_customers)


@dataclass(frozen=True)
class WorstRemove:
    # removes points with highest cost (hoping that repair
    # operators will find a better place to emplace them)
    route_cost: Callable[[cvrp.Route], float]
    cfg: BasicDestroyConfig

    def __call__(self, sol: cvrp.Solution):
        dim = self.cfg.problem.dim
        savings = np.empty(dim, dtype=float)

        for route in sol.routes:
            route_cost = self.route_cost(route)
            route_copy = np.asarray(route)
            for customer in route:
                r = route_copy[route_copy != customer]
                saving = route_cost - 0 if len(r) == 0 else self.route_cost(r)
                savings[customer] = saving

        n = self.cfg.nodes_to_remove()
        worst_nodes = -savings[1:].argsort()[:n]
        return remove_unassigned(sol, set(worst_nodes))


@dataclass(frozen=True)
class SubstringRemoval:
    max_substring_removals: int
    max_string_size: int
    cfg: BasicDestroyConfig

    def __call__(self, sol: cvrp.Solution):
        avg_route_size = int(np.mean([len(r) for r in sol.routes]))
        max_string_size = max(self.max_string_size, avg_route_size)
        max_substring_removals = min(len(sol.routes), self.max_substring_removals)

        destroyed_routes = set()
        unassigned = set()
        center = self.cfg.rng.integers(1, self.cfg.problem.dim)
        distances = self.cfg.problem.distances

        for customer in construct.neighbours(center, distances):
            if len(destroyed_routes) >= max_substring_removals:
                break

            if customer in unassigned:
                continue

            route = sol.find_route(customer)
            route_idx = sol.routes.index(route)

            if route_idx in destroyed_routes:
                continue

            removed = self.remove_substring(route, customer, max_string_size)
            unassigned.update(removed)
            destroyed_routes.add(route_idx)

        return remove_unassigned(sol, unassigned)

    def remove_substring(
        self,
        route: cvrp.Route,
        customer: int,
        max_string_size: int,
    ) -> cvrp.Unassigned:
        size = self.cfg.rng.integers(1, min(len(route), max_string_size) + 1)
        start = route.index(customer) - self.cfg.rng.integers(0, size)

        indices = np.arange(start, start + size) % len(route)
        removed = {route[i] for i in indices}
        return removed


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
            partial_sol.unassigned,
            count=len(partial_sol.unassigned),
            dtype=np.int32,
        )
        self.cfg.rng.shuffle(unassigned)
        routes = partial_sol.routes
        # shorthands
        min_vehicles = self.cfg.problem.min_vehicles
        assign = len(unassigned)

        for i, customer in enumerate(unassigned):
            route, idx = self.best_insert(customer, routes)
            if route is None or len(routes) + assign - i <= min_vehicles:
                # capacity constraints can't be satisfied
                # with current number of vehicles,
                # create a new route with single customer
                # OR insufficient number of vahicles, need to add one
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
        route_loads = tuple(self.cfg.problem.route_load(r) for r in routes)
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
