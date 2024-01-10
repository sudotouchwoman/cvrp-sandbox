from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count, takewhile
import time
from typing import Generator, Optional, Sequence

import numpy as np
from rich import progress

from . import operators, cvrp, accept, get_logger


logger = get_logger(__name__)

# https://www.researchgate.net/publication/346576408_An_Adaptive_Large_Neighborhood_Search_for_the_Larger-Scale_Instances_of_Green_Vehicle_Routing_Problem_with_Time_Windows
# https://github.com/N-Wouda/ALNS


@dataclass(slots=True)
class Timer:
    start: float = field(default_factory=time.time)
    end: float = None

    @property
    def duration(self):
        return self.end - self.start


@contextmanager
def scoped_timer() -> Generator[Timer, None, None]:
    try:
        t = Timer()
        yield t
    finally:
        t.end = time.time()


@dataclass(frozen=True)
class TracedSolution:
    best_solution: cvrp.Solution
    iteration_costs: Sequence[float]
    best_costs: Sequence[float]
    elapsed_time: float
    iterations: int


@dataclass(frozen=True)
class ALNS:
    accept: accept.AcceptanceCriterion
    destroy_operators: Sequence[operators.DestroyOperator] = field(default_factory=list)
    repair_operators: Sequence[operators.RepairOperator] = field(default_factory=list)

    def iterate(
        self,
        initial_solution: cvrp.Solution,
        max_iter: Optional[int] = None,
        max_runtime: float = 60,
        seed: int = 42,
        verbose: bool = False,
    ) -> TracedSolution:
        # to make `iterate` reentrant, create method-scoped rng
        rng = np.random.default_rng(seed=seed)
        elapsed = 0.0

        best_sol = current_sol = initial_solution
        iteration_costs = []
        best_costs = []

        try:
            for i in progress.track(
                takewhile(lambda x: not max_iter or x < max_iter, count()),
                disable=not verbose,
                description="Running ALNS CVRP optimization",
                total=max_iter,
            ):
                with scoped_timer() as timer:
                    destroyed = self.destroy(rng)(current_sol)
                    candidate = self.repair(rng)(destroyed)
                    if candidate is None:
                        # failed to reconstruct solution from partial solution
                        # after destroy operator: usually this means we are
                        # out of vehicles, I guess?
                        logger.warning(f"it: {i}, failed to repair solution, skipping iteration")
                        continue

                    if self.accept(
                        rng,
                        best_sol.cost,
                        current_sol.cost,
                        candidate.cost,
                    ):
                        current_sol = candidate

                    if best_sol.cost > candidate.cost:
                        best_sol = candidate

                iteration_costs.append(current_sol.cost)
                best_costs.append(best_sol.cost)

                elapsed += timer.duration
                if elapsed > max_runtime:
                    logger.info("max_runtime exceeded, stopping...")
                    break

        except KeyboardInterrupt:
            logger.info("interrupted, return best solution so far...")

        return TracedSolution(
            best_solution=best_sol,
            iteration_costs=iteration_costs,
            best_costs=best_costs,
            elapsed_time=elapsed,
            iterations=i,
        )

    def destroy(self, rng) -> operators.DestroyOperator:
        return rng.choice(self.destroy_operators)

    def repair(self, rng) -> operators.RepairOperator:
        return rng.choice(self.repair_operators)
