from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

try:
    from .evaluation import AON, QUADRATIC, hypergraph_cut, is_balanced
    from .loader import hyperedge_weights, incidence_to_hyperedges
except ImportError:
    from evaluation import AON, QUADRATIC, hypergraph_cut, is_balanced  # type: ignore
    from loader import hyperedge_weights, incidence_to_hyperedges  # type: ignore


CONFIG_DIR = Path(__file__).resolve().parent
KAHYPAR_CONFIG = CONFIG_DIR / "cut_kKaHyPar_sea20.ini"


def run_exhaustive_search(
    H: np.ndarray,
    k: int = 2,
    objective: str = "aon",
    return_best_score: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    if incidence.shape[1] == 0:
        raise ValueError("Hypergraph must contain at least one vertex")
    if objective == QUADRATIC and k != 2:
        raise ValueError("The quadratic objective is only supported for k=2 in exhaustive search")

    best_score = float("inf")
    best_partition = None

    for rest in product(range(k), repeat=incidence.shape[1] - 1):
        partition = np.array((0, *rest), dtype=int)
        if not is_balanced(partition, k):
            continue
        score = hypergraph_cut(partition, incidence, objective=objective)
        if score < best_score:
            best_score = score
            best_partition = partition

    if best_partition is None:
        raise RuntimeError("No balanced partition found")

    if return_best_score:
        return best_partition, float(best_score)
    return best_partition


def run_kahypar(
    H: np.ndarray,
    k: int = 2,
    objective: str = AON,
    imbalance: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    try:
        import kahypar  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "KaHyPar's Python module is not installed in this environment. "
            "Install `kahypar` to use this baseline."
        ) from exc

    if objective != AON:
        raise ValueError(f"Unsupported KaHyPar objective '{objective}'. Only '{AON}' is supported.")
    if not KAHYPAR_CONFIG.exists():
        raise RuntimeError(f"KaHyPar config file not found: {KAHYPAR_CONFIG}")

    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    num_hyperedges, num_vertices = incidence.shape
    hyperedges = incidence_to_hyperedges(incidence)
    weights = hyperedge_weights(incidence)
    hyperedge_indices = [0]
    flat_hyperedges: list[int] = []
    for hyperedge in hyperedges:
        flat_hyperedges.extend(hyperedge)
        hyperedge_indices.append(len(flat_hyperedges))

    context = kahypar.Context()
    context.loadINIconfiguration(str(KAHYPAR_CONFIG))
    context.setK(k)
    context.setEpsilon(float(imbalance))
    context.suppressOutput(not verbose)

    hypergraph_handle = kahypar.Hypergraph(
        num_vertices,
        num_hyperedges,
        hyperedge_indices,
        flat_hyperedges,
        k,
        weights.astype(int).tolist(),
        np.ones(num_vertices, dtype=int).tolist(),
    )
    kahypar.partition(hypergraph_handle, context)

    return np.array([hypergraph_handle.blockID(vertex) for vertex in range(num_vertices)], dtype=int)
