from __future__ import annotations

import dimod
import numpy as np
from dimod import SimulatedAnnealingSampler
# from dwave.samplers import PathIntegralAnnealingSampler

try:
    from .utils import incidence_to_hyperedges
except ImportError:
    from utils import incidence_to_hyperedges  # type: ignore


def _build_aon_bqm_from_hyperedges(hyperedges, lam):
    for index, edge in enumerate(hyperedges):
        if len(edge) > 3:
            raise ValueError(
                f"Hyperedge {index} has size {len(edge)} > 3. "
                "The AoN objective is only quadratic for k <= 3."
            )

    all_nodes = [node for edge in hyperedges for node in edge]
    if not all_nodes:
        raise ValueError("hyperedges contains no vertices; cannot infer num_vertices")

    num_vertices = max(all_nodes) + 1
    linear = {i: 0.0 for i in range(num_vertices)}
    quadratic = {}
    offset = 0.0

    # AoN cut term
    for edge in hyperedges:
        k = len(edge)
        if k <= 1:
            continue

        if k == 2:
            i, j = edge[0], edge[1]
            linear[i] += 1.0
            linear[j] += 1.0
            pair = (min(i, j), max(i, j))
            quadratic[pair] = quadratic.get(pair, 0.0) - 2.0

        elif k == 3:
            i, j, l = edge[0], edge[1], edge[2]
            linear[i] += 1.0
            linear[j] += 1.0
            linear[l] += 1.0
            for a, b in ((i, j), (i, l), (j, l)):
                pair = (min(a, b), max(a, b))
                quadratic[pair] = quadratic.get(pair, 0.0) - 1.0

    # balance term
    for i in range(num_vertices):
        linear[i] += lam * (1 - num_vertices)

    # quadratic: same nested index loop over all unordered vertex pairs
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            pair = (i, j)
            quadratic[pair] = quadratic.get(pair, 0.0) + 2.0 * lam

    offset += lam * (num_vertices ** 2) / 4.0

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype="BINARY")


def run_hypergraph_simulated_annealing(
    incidence_matrix,
    k,
    lam=1.0,
    num_reads=100,
    seed=None,
):
    incidence = np.asarray(incidence_matrix, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    if k != 2:
        raise ValueError("This binary QUBO simulated annealer currently only supports k=2")

    hyperedges, _ = incidence_to_hyperedges(incidence.T)
    bqm = _build_aon_bqm_from_hyperedges(hyperedges, lam)

    sampler = SimulatedAnnealingSampler()
    # sampler = PathIntegralAnnealingSampler()
    sample_kwargs = {"num_reads": num_reads}
    if seed is not None:
        sample_kwargs["seed"] = seed
    response = sampler.sample(bqm, **sample_kwargs)
    best_sample = response.first.sample
    return np.array([best_sample[i] for i in range(incidence.shape[1])], dtype=int)
