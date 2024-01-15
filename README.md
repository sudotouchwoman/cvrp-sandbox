# **CVRP primer with ALNS**

Repository contains implementation of ALNS metaheuristic for
Capacitated Vehicle Routing Problem (CVRP) solution. Adaptive Large Neighbourhood Search
is a metaheuristic that combines ideas of Local Search (LS) heuristics and searches
optimal solutions in much larger neighbourhoods by sequentially *destroying* part of
current solution and trying to *repair* it. For that purpose, ALNS may utilize one of
several preconfigured *destroy* and *repair* *operators*, i.e. heuristics that either unassign customers
from their respective routes or try to fit them into a (presumably) better positions in
possibly different routes.

To faciliate divergence in produced solutions and adapt solver to a concrete problem, on each
iteration both types of operators are picked at random (customarily, using a roulette wheel).
Essentially, that is where adaptive in ALNS stems from. In this implementation, however, as of now,
relative operator weights are not fitted during optimization.

One can get more details about the algorithm in the literature:

+ [Large Neighbourhood Method proposal by P.Shaw](https://link.springer.com/chapter/10.1007/3-540-49481-2_30)
+ [LNS paper by S.Ropke](https://link.springer.com/chapter/10.1007/978-1-4419-1665-5_13)
+ [Comparison of acceptance criteria for ALNS metaheuristic](https://link.springer.com/article/10.1007/s10732-018-9377-x)
+ [ALNS applied to GVRPTW](https://www.researchgate.net/publication/346576408_An_Adaptive_Large_Neighborhood_Search_for_the_Larger-Scale_Instances_of_Green_Vehicle_Routing_Problem_with_Time_Windows)

Related repositories that strongly influenced the implementation:

+ [Related Python package with great docs](https://alns.readthedocs.io/en/latest/)
+ [Decomposition Strategies for VRP](https://github.com/alberto-santini/cvrp-decomposition)
+ [C++ implementation, loosely based on original implementation by S.Ropke](https://github.com/alberto-santini/adaptive-large-neighbourhood-search)

## **Usage**

```sh
pip install -r requirements.txt
mkdir -p data
cd notebooks
# will evaluate ALNS solver on problems from cvrplib
# and produce benchmark.csv file in data/ directory
# with results and metadata
python3 benchmark.py
```

Notebooks under `notebooks` directory serve for visualization of results.
