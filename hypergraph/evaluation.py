from __future__ import annotations

import numpy as np

try:
    from .loader import hyperedge_weights
except ImportError:
    from loader import hyperedge_weights  # type: ignore


AON = "aon"
KM1 = "km1"
QUADRATIC = "quadratic"
SUPPORTED_OBJECTIVES = {AON, KM1, QUADRATIC}


def is_balanced(partition, k: int) -> bool:
    block_sizes = np.bincount(np.asarray(partition, dtype=int), minlength=k)
    return block_sizes.max() - block_sizes.min() <= 1


def hyperedge_connectivity(partition, hyperedge) -> int:
    return len({int(partition[vertex]) for vertex in hyperedge})


def hypergraph_cut(partition, H: np.ndarray, objective: str = AON) -> float:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective '{objective}'. Expected one of {sorted(SUPPORTED_OBJECTIVES)}")

    labels = np.asarray(partition, dtype=int)
    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    if len(labels) != incidence.shape[1]:
        raise ValueError("Partition length does not match the number of vertices")
    if objective == QUADRATIC and len(set(labels.tolist())) > 2:
        raise ValueError("The quadratic objective is only defined here for bipartitions (k=2)")

    weights = hyperedge_weights(incidence)
    total = 0.0
    for row, weight in zip(incidence, weights, strict=True):
        hyperedge = np.flatnonzero(row)
        blocks_touched = hyperedge_connectivity(labels, hyperedge)
        if blocks_touched <= 1:
            continue
        if objective == AON:
            total += weight
        elif objective == KM1:
            total += weight * float(blocks_touched - 1)
        elif objective == QUADRATIC:
            block_counts = np.bincount(labels[hyperedge], minlength=2)
            total += weight * float(block_counts[0] * block_counts[1])
    return float(total)


def evaluate_partition(partition, H: np.ndarray, k: int = 2, verbose: bool = True) -> dict[str, object]:
    labels = np.asarray(partition, dtype=int)
    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    if len(labels) != incidence.shape[1]:
        raise ValueError("Partition length does not match the number of vertices")

    observed_k = len(set(labels.tolist()))
    if observed_k > k:
        raise ValueError(f"Partition uses {observed_k} blocks, but k={k}")

    block_sizes = np.bincount(labels, minlength=k)
    metrics = {
        "aon": hypergraph_cut(labels, incidence, objective=AON),
        "km1": hypergraph_cut(labels, incidence, objective=KM1),
        "quadratic": hypergraph_cut(labels, incidence, objective=QUADRATIC) if k == 2 else None,
        "balanced": bool(is_balanced(labels, k)),
        "block_sizes": block_sizes,
    }

    if verbose:
        print(f"AON objective: {metrics['aon']}")
        print(f"KM1 objective: {metrics['km1']}")
        if metrics["quadratic"] is not None:
            print(f"Quadratic objective: {metrics['quadratic']}")
        print(f"Block sizes: {block_sizes}")
        print("Partition is balanced" if metrics["balanced"] else "Partition is imbalanced")

    return metrics
