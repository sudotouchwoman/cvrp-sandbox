from dataclasses import dataclass
from typing import Literal, Protocol, Union
import numpy as np


class AcceptanceCriterion(Protocol):
    def __call__(
        self,
        rng: np.random.Generator,
        best: float,
        current: float,
        candidate: float,
    ) -> bool:
        ...


DecayMethod = Union[Literal["linear"], Literal["exponential"]]


def __decay(p: float, step: float, method: DecayMethod):
    if method == "linear":
        return p - step

    if method == "exponential":
        return p * step

    raise ValueError(f"unknown method: {method}")


def always_accept(
    rng: np.random.Generator,
    best: float,
    current: float,
    candidate: float,
) -> bool:
    return True


def hill_climb(
    rng: np.random.Generator,
    best: float,
    current: float,
    candidate: float,
) -> bool:
    return candidate < current


@dataclass
class RandomAccept:
    proba: float
    end_proba: float
    step: float
    method: DecayMethod

    def __call__(
        self,
        rng: np.random.Generator,
        best: float,
        current: float,
        candidate: float,
    ):
        res = candidate < current
        if not res:
            res = rng.random() <= self.proba

        decayed = __decay(self.proba, self.step, self.method)
        self.proba = max(self.end_proba, decayed)
        return res


@dataclass
class SimulatedAnnealing:
    temperature: float
    temperature_end: float
    step: float
    method: DecayMethod

    def __call__(
        self,
        rng: np.random.Generator,
        best: float,
        current: float,
        candidate: float,
    ):
        proba = np.exp((current - candidate) / self.temperature)

        self.temperature = max(
            self.temperature_end,
            __decay(self.temperature, self.step, self.method),
        )

        return rng.random() <= proba

    @classmethod
    def fit(
        cls,
        init_cost: float,
        worse: float,
        accept_proba: float,
        num_iters: int,
        method: DecayMethod,
    ):
        if method not in ("linear", "exponential"):
            raise ValueError(f"unknown decay method: {method}")

        # TODO n.teterin: add argument validation

        start_temperature = -worse * init_cost / np.log(accept_proba)
        if method == "linear":
            step = (start_temperature - 1) / num_iters
        if method == "exponential":
            step = (1 / start_temperature) ** (1 / num_iters)

        return cls(start_temperature, 1, step=step, method=method)
