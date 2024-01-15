#! /usr/bin/env python3
import logging
import os
import vrplib
import pathlib

import pandas as pd
import numpy as np

from concurrent import futures
from functools import partial

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

from urllib import error

import lns
from lns import cvrp


logger = lns.get_logger(__name__)


def download_problem(instance_name: str, root: pathlib.Path):
    try:
        instance_path = root / f"{instance_name}.vrp"
        solution_path = root / f"{instance_name}.sol"

        if not instance_path.is_file():
            vrplib.download_instance(instance_name, instance_path)

        if not solution_path.is_file():
            vrplib.download_solution(instance_name, solution_path)

    except error.HTTPError as e:
        logger.warning(f"{e}: {instance_name}")


def read_problem(name: str, root: pathlib.Path):
    instance_path = root / f"{name}.vrp"
    solution_path = root / f"{name}.sol"

    if not instance_path.is_file():
        vrplib.download_instance(name, instance_path)
    if not solution_path.is_file():
        vrplib.download_solution(name, solution_path)

    data = vrplib.read_instance(instance_path)
    bks = vrplib.read_solution(solution_path)

    problem = cvrp.Problem.from_vrplib(data)
    opt_sol = cvrp.Solution.from_vrplib(bks)

    return name, problem, opt_sol


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    problem: cvrp.Problem
    opt_solution: cvrp.Solution
    initial_solution: cvrp.Solution
    alns_solution: lns.alns.TracedSolution

    @cached_property
    def mape(self):
        best = self.alns_solution.best_solution.cost
        optimal = self.opt_solution.cost
        return (best - optimal) / optimal

    @cached_property
    def elapsed(self):
        return self.alns_solution.elapsed_time

    @cached_property
    def mape_init(self):
        init = self.initial_solution.cost
        optimal = self.opt_solution.cost
        return (init - optimal) / optimal


# evaluates given model and initial solution builder
# on provided problem. name and optimal solution are also
# saved to the result for convenience
def benchmark_model(
    name: str,
    p: cvrp.Problem,
    opt_sol: cvrp.Solution,
    model: Callable[[cvrp.Problem, cvrp.Solution], lns.alns.TracedSolution],
    builder: lns.construct.SolutionConstructor,
):
    logger.debug(f"Working on {p.name}")
    initial_sol = builder(p)
    best_sol = model(p, initial_sol)

    if not cvrp.check_solution(p, best_sol.best_solution):
        logger.error(f"Solution consistency check failed: {p.name}")

    result = BenchmarkResult(name, p, opt_sol, initial_sol, best_sol)

    mape = result.mape * 100
    init_mape = result.mape_init * 100

    # display some info
    best = best_sol.best_solution.cost
    optimal = opt_sol.cost

    logger.debug(f"[{best:.3f}/{optimal}] MAPE: {mape:.3f}%, initial: {init_mape:.3f}%")
    logger.debug(f"done in {best_sol.elapsed_time:.3}s")

    return result


def better_alns_factory(
    seed: int = 10, max_iterations: int = 10_000, max_runtime: float = 60
):
    # ensure reentrance
    rng = np.random.default_rng(seed=seed)

    def solve(p: cvrp.Problem, initial: cvrp.Solution):
        accept_criterion = lns.accept.SimulatedAnnealing.fit(
            initial.cost,
            worse=0.5,
            accept_proba=0.1,
            num_iters=max_iterations,
            method="exponential",
        )

        cfg = lns.operators.BasicDestroyConfig(
            problem=p,
            bounds=[min(5, 0.1 * p.dim), min(50, 0.5 * p.dim)],
            rng=rng,
        )

        destroy_operators = [
            lns.operators.RandomRemove(cfg),
            lns.operators.SubstringRemoval(
                max_substring_removals=2,
                max_string_size=12,
                cfg=cfg,
            ),
        ]

        repair_operators = [
            lns.operators.GreedyRepair(
                lns.operators.BasicRepairConfig(
                    problem=p,
                    rng=rng,
                )
            )
        ]

        solver = lns.alns.ALNS(
            accept=accept_criterion,
            destroy_operators=destroy_operators,
            repair_operators=repair_operators,
        )

        return solver.iterate(
            initial,
            max_iter=max_iterations,
            max_runtime=max_runtime,
            verbose=True,
            handle_interrupts=False,
        )

    return solve


def alns_factory(seed: int = 10, max_iterations: int = 10_000, max_runtime: float = 60):
    # ensure reentrance
    rng = np.random.default_rng(seed=seed)

    def solve(p: cvrp.Problem, initial: cvrp.Solution):
        bounds = [max(1, int(0.05 * p.dim)), max(3, int(0.4 * p.dim))]

        accept_criterion = lns.accept.RandomAccept(
            proba=0.5,
            end_proba=1e-2,
            step=0.99,
            method="exponential",
        )

        destroy_operators = [
            lns.operators.RandomRemove(
                lns.operators.BasicDestroyConfig(
                    problem=p,
                    bounds=bounds,
                    rng=rng,
                )
            )
        ]

        repair_operators = [
            lns.operators.GreedyRepair(
                lns.operators.BasicRepairConfig(
                    problem=p,
                    rng=rng,
                )
            )
        ]

        solver = lns.alns.ALNS(
            accept=accept_criterion,
            destroy_operators=destroy_operators,
            repair_operators=repair_operators,
        )

        return solver.iterate(
            initial,
            max_iter=max_iterations,
            max_runtime=max_runtime,
            verbose=False,
            handle_interrupts=False,
        )

    return solve


def main():
    # use several problem sets: I added F and M since they contain
    # some big instances (up to 200 clients) with known optinal solutions
    problem_sets = {
        "A": [n for n in vrplib.list_names() if n.startswith("A-")],
        "B": [n for n in vrplib.list_names() if n.startswith("B-")],
        "E": [n for n in vrplib.list_names() if n.startswith("E-")],
        "F": [n for n in vrplib.list_names() if n.startswith("F-")],
        "M": [n for n in vrplib.list_names() if n.startswith("M-")],
    }

    data_dir = pathlib.Path("../data/")
    problems = [i for problem_set in problem_sets.values() for i in problem_set]

    logger.debug("ensuring that instance/solution files exist")
    with futures.ThreadPoolExecutor() as executor:
        for _ in executor.map(partial(download_problem, root=data_dir), problems):
            ...

    logger.info(
        f"evaluating on {len(problems)} instances from {len(problem_sets)} sets"
    )

    benchmark_results = {}
    for key, problems in problem_sets.items():
        results = tuple(
            benchmark_model(
                name,
                p,
                opt,
                better_alns_factory(max_iterations=100_000, max_runtime=300),
                lns.construct.nearest_neighbour_builder,
            )
            for name, p, opt in map(
                lambda name: read_problem(name, root=data_dir), problems
            )
        )
        benchmark_results[key] = results

    # convert to DataFrame and save as csv file
    with_names = [(subset, b) for subset, bs in benchmark_results.items() for b in bs]
    df = pd.DataFrame(
        {
            "subset": [n for n, _ in with_names],
            "time": [b.elapsed for _, b in with_names],
            "iterations": [b.alns_solution.iterations for _, b in with_names],
            "optimal-score": [b.opt_solution.cost for _, b in with_names],
            "best-score": [b.alns_solution.best_solution.cost for _, b in with_names],
            "init-score": [b.initial_solution.cost for _, b in with_names],
            "mape-best": [b.mape for _, b in with_names],
            "mape-initial": [b.mape_init for _, b in with_names],
            "dimension": [b.problem.dim for _, b in with_names],
            "min-vehicles": [b.problem.min_vehicles for _, b in with_names],
            "full-name": [b.problem.name for _, b in with_names],
            "name": [b.name for _, b in with_names],
        }
    )

    # to lazy to wrap this into click for now
    export_path = os.getenv("CSV_EXPORT_PATH", "../data/benchmark.csv")
    logger.info(f"exporting results to {export_path}")
    df.to_csv(export_path, index=False)


if __name__ == "__main__":
    main()
