from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# Build the primal graph used to test whether a hypergraph is connected.
def _primal_graph(num_vertices: int, hyperedges: tuple[tuple[int, ...], ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    for hyperedge in hyperedges:
        for u, v in combinations(hyperedge, 2):
            graph.add_edge(u, v)
    return graph


# Convert an incidence matrix into explicit hyperedge vertex tuples.
def incidence_to_hyperedges(H: np.ndarray) -> tuple[tuple[int, ...], ...]:
    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    return tuple(tuple(np.flatnonzero(row).tolist()) for row in incidence)


# Extract one weight per hyperedge from the shared nonzero row values.
def hyperedge_weights(H: np.ndarray) -> np.ndarray:
    incidence = np.asarray(H)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    weights = []
    for row in incidence:
        nonzero = row[row != 0]
        if len(nonzero) == 0:
            weights.append(0.0)
            continue
        first_weight = float(nonzero[0])
        if not np.all(nonzero == first_weight):
            raise ValueError("Each hyperedge row must use the same non-zero value for all incident vertices")
        weights.append(first_weight)
    return np.asarray(weights, dtype=float)


# Return the common hyperedge size when the hypergraph is uniform.
def hyperedge_size(H: np.ndarray) -> int | None:
    incidence = np.asarray(H, dtype=int)
    sizes = np.count_nonzero(incidence, axis=1)
    if len(sizes) == 0:
        return None
    size = int(sizes[0])
    if np.all(sizes == size):
        return size
    return None


# Compute the average number of incident hyperedges per vertex.
def average_node_degree(H: np.ndarray) -> float:
    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    if incidence.shape[1] == 0:
        raise ValueError("Hypergraph must contain at least one vertex")
    return float(np.count_nonzero(incidence) / incidence.shape[1])


# Build a weighted incidence matrix from sampled hyperedges.
def _build_incidence_matrix(
    n: int,
    hyperedges: tuple[tuple[int, ...], ...],
    rng: np.random.Generator,
    weight_range: tuple[int, int] | None,
) -> np.ndarray:
    H = np.zeros((len(hyperedges), n), dtype=int)
    for edge_index, hyperedge in enumerate(hyperedges):
        if weight_range is None:
            weight = 1
        else:
            weight = int(rng.integers(weight_range[0], weight_range[1] + 1))
        H[edge_index, list(hyperedge)] = weight
    return H


# Validate that a weight range is either absent or a valid positive interval.
def _validate_weight_range(weight_range: tuple[int, int] | None) -> None:
    if weight_range is None:
        return
    min_weight, max_weight = weight_range
    if min_weight <= 0 or max_weight < min_weight:
        raise ValueError("weight_range must contain positive integers with min <= max")


# Validate that a node-count range is well formed.
def _validate_n_range(n_range: tuple[int, int]) -> None:
    n_min, n_max = n_range
    if n_min < 2 or n_max < n_min:
        raise ValueError("n_range must satisfy 2 <= min <= max")


# Format a compact filename tag such as n9-12 or w1-10.
def _range_tag(label: str, value_range: tuple[int, int]) -> str:
    start, end = value_range
    return f"{label}{start}-{end}"


def _value_tag(label: str, value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value_str = str(value).replace(".", "p")
    return f"{label}{value_str}"


# Format a rank tag for either one edge size or a size range.
def _rank_tag(edge_sizes: int | tuple[int, ...]) -> str:
    if isinstance(edge_sizes, int):
        return f"r{edge_sizes}"
    normalized = tuple(sorted(set(int(size) for size in edge_sizes)))
    if len(normalized) == 1:
        return f"r{normalized[0]}"
    return f"r{normalized[0]}-{normalized[-1]}"


# Build the shared filename prefix for a family of hypergraph instances.
def hypergraph_group_prefix(
    edge_sizes: int | tuple[int, ...],
    n_range: tuple[int, int],
    weight_range: tuple[int, int] | None = None,
    label: str | None = None,
) -> str:
    _validate_n_range(n_range)
    _validate_weight_range(weight_range)

    tags = ["hypergraph"]
    if label:
        tags.append(label)
    tags.append(_rank_tag(edge_sizes))
    tags.append(_range_tag("n", n_range))
    tags.append("w1" if weight_range is None else _range_tag("w", weight_range))
    return "_".join(tags)


# Generate a random mixed-size hypergraph with optional weights.
def generate_random_mixed_hypergraph(
    n: int = 10,
    edge_sizes: tuple[int, ...] = (3, 4, 5),
    num_edges: int | None = None,
    weight_range: tuple[int, int] | None = None,
    seed: int | None = None,
    require_vertex_coverage: bool = True,
    require_connected: bool = True,
    max_tries: int = 1_000,
) -> np.ndarray:
    size_choices = tuple(sorted(set(int(size) for size in edge_sizes)))
    if not size_choices:
        raise ValueError("edge_sizes must contain at least one size")
    if min(size_choices) < 2:
        raise ValueError("All hyperedge sizes must be at least 2")
    if max(size_choices) > n:
        raise ValueError("Hyperedge sizes cannot exceed the number of vertices")
    _validate_weight_range(weight_range)

    candidate_edges_by_size = {
        size: list(combinations(range(n), size))
        for size in size_choices
    }
    total_possible_edges = sum(len(edges) for edges in candidate_edges_by_size.values())
    if num_edges is None or not (1 <= num_edges <= total_possible_edges):
        raise ValueError("num_edges must be between 1 and the number of possible mixed-size hyperedges")

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        remaining_edges = {
            size: list(edges)
            for size, edges in candidate_edges_by_size.items()
        }
        for edges in remaining_edges.values():
            rng.shuffle(edges)

        hyperedges_list = []
        while len(hyperedges_list) < num_edges:
            available_sizes = [size for size, edges in remaining_edges.items() if edges]
            if not available_sizes:
                break
            size = int(rng.choice(available_sizes))
            hyperedges_list.append(remaining_edges[size].pop())

        hyperedges = tuple(sorted(hyperedges_list))
        if len(hyperedges) != num_edges:
            continue

        if require_vertex_coverage:
            covered_vertices = set().union(*hyperedges)
            if len(covered_vertices) < n:
                continue

        if require_connected:
            primal_graph = _primal_graph(n, hyperedges)
            if not nx.is_connected(primal_graph):
                continue

        return _build_incidence_matrix(n, hyperedges, rng, weight_range)

    raise RuntimeError(
        "Failed to generate a random mixed-size hypergraph with the requested constraints. "
        "Try reducing num_edges or disabling connectivity checks."
    )


# Generate a random uniform hypergraph with optional weights.
def generate_random_uniform_hypergraph(
    n: int = 10,
    edge_size: int = 3,
    edge_probability: float = 0.2,
    num_edges: int | None = None,
    weight_range: tuple[int, int] | None = None,
    seed: int | None = None,
    require_vertex_coverage: bool = True,
    require_connected: bool = True,
    max_tries: int = 1_000,
) -> np.ndarray:
    if edge_size < 2:
        raise ValueError("edge_size must be at least 2")
    if edge_size > n:
        raise ValueError("edge_size cannot exceed the number of vertices")
    if num_edges is None and not (0.0 < edge_probability <= 1.0):
        raise ValueError("edge_probability must be in (0, 1] when num_edges is not provided")
    _validate_weight_range(weight_range)

    all_edges = list(combinations(range(n), edge_size))
    if num_edges is not None and not (1 <= num_edges <= len(all_edges)):
        raise ValueError("num_edges must be between 1 and the number of possible uniform hyperedges")

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        if num_edges is not None:
            selected_ids = rng.choice(len(all_edges), size=num_edges, replace=False)
            hyperedges = tuple(all_edges[index] for index in sorted(selected_ids))
        else:
            mask = rng.random(len(all_edges)) < edge_probability
            hyperedges = tuple(edge for edge, keep in zip(all_edges, mask, strict=True) if keep)

        if not hyperedges:
            continue

        if require_vertex_coverage:
            covered_vertices = set().union(*hyperedges)
            if len(covered_vertices) < n:
                continue

        if require_connected:
            primal_graph = _primal_graph(n, hyperedges)
            if not nx.is_connected(primal_graph):
                continue

        return _build_incidence_matrix(n, hyperedges, rng, weight_range)

    raise RuntimeError(
        "Failed to generate a random uniform hypergraph with the requested constraints. "
        "Try increasing edge_probability, setting num_edges explicitly, or disabling connectivity checks."
    )


# Save one hypergraph incidence matrix into the data directory.
def save_hypergraph(H: np.ndarray, name: str) -> Path:
    incidence = np.asarray(H, dtype=int)
    if incidence.ndim != 2:
        raise ValueError("Incidence matrix must be 2-dimensional")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.npy"
    np.save(path, incidence)
    return path


# Load one hypergraph incidence matrix from the data directory.
def load_hypergraph(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.npy"
    return np.load(path)


# List stored hypergraphs, optionally filtering to a specific uniform rank.
def list_hypergraph_instances(edge_size: int | None = None) -> list[str]:
    names = sorted(path.stem for path in DATA_DIR.glob("hypergraph_*.npy"))
    if edge_size is None:
        return names
    rank_tag = f"_r{edge_size}_"
    legacy_prefix = f"hypergraph_r{edge_size}_"
    return [name for name in names if name.startswith(legacy_prefix) or rank_tag in name]


# List all stored hypergraph names that share a common filename prefix.
def list_hypergraph_group(prefix: str) -> list[str]:
    return sorted(path.stem for path in DATA_DIR.glob(f"{prefix}*.npy"))


# Load all stored hypergraphs that share a common filename prefix.
def load_hypergraph_group(prefix: str) -> dict[str, np.ndarray]:
    return {name: load_hypergraph(name) for name in list_hypergraph_group(prefix)}


# Generate and store a named family of uniform hypergraph instances.
def generate_and_store_uniform_hypergraph_group(
    edge_size: int,
    count: int = 10,
    seed: int = 0,
    n_range: tuple[int, int] = (9, 12),
    weight_range: tuple[int, int] | None = None,
    average_degree: float = 5.0,
    degree_jitter: int = 1,
    label: str | None = None,
) -> list[Path]:
    _validate_n_range(n_range)
    rng = np.random.default_rng(seed)
    saved_paths: list[Path] = []
    prefix = hypergraph_group_prefix(edge_size, n_range, weight_range, label=label)
    for index in range(count):
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        num_edges = max(1, int(round(average_degree * n / edge_size)))
        if degree_jitter > 0:
            num_edges += int(rng.integers(-degree_jitter, degree_jitter + 1))
            num_edges = max(1, num_edges)
        num_possible_edges = math.comb(n, edge_size)
        num_edges = min(num_edges, num_possible_edges)
        H = generate_random_uniform_hypergraph(
            n=n,
            edge_size=edge_size,
            num_edges=num_edges,
            weight_range=weight_range,
            seed=seed * 1_000 + index,
        )
        name = f"{prefix}_{index}"
        saved_paths.append(save_hypergraph(H, name))
    return saved_paths


def generate_and_store_connected_uniform_hypergraphs_for_n(
    n: int,
    count: int = 10,
    seed: int = 0,
    edge_size: int = 3,
    weight_range: tuple[int, int] | None = None,
    average_degree: float = 3.0,
    degree_jitter: int = 1,
) -> list[Path]:
    if n < 2:
        raise ValueError("n must be at least 2")
    if edge_size < 2:
        raise ValueError("edge_size must be at least 2")
    if edge_size > n:
        raise ValueError("edge_size cannot exceed n")
    if average_degree <= 0:
        raise ValueError("average_degree must be positive")
    if degree_jitter < 0:
        raise ValueError("degree_jitter must be non-negative")
    _validate_weight_range(weight_range)

    rng = np.random.default_rng(seed)
    saved_paths: list[Path] = []
    prefix_parts = [
        "hypergraph",
        _rank_tag(edge_size),
        _value_tag("n", n),
        _value_tag("d", average_degree),
        "w1" if weight_range is None else _range_tag("w", weight_range),
    ]
    prefix = "_".join(prefix_parts)

    for index in range(count):
        num_edges = max(1, int(round(average_degree * n / edge_size)))
        if degree_jitter > 0:
            num_edges += int(rng.integers(-degree_jitter, degree_jitter + 1))
            num_edges = max(1, num_edges)
        num_possible_edges = math.comb(n, edge_size)
        num_edges = min(num_edges, num_possible_edges)
        H = generate_random_uniform_hypergraph(
            n=n,
            edge_size=edge_size,
            num_edges=num_edges,
            weight_range=weight_range,
            seed=seed * 1_000 + index,
            require_vertex_coverage=True,
            require_connected=True,
        )
        name = f"{prefix}_{index}"
        saved_paths.append(save_hypergraph(H, name))

    return saved_paths


# Generate and store a default small benchmark set of unweighted 3/4-uniform hypergraphs.
def generate_and_store_small_unweighted_uniform_hypergraphs(
    count: int = 10,
    edge_sizes: tuple[int, ...] = (3, 4),
    seed: int = 0,
    n_range: tuple[int, int] = (9, 12),
) -> list[Path]:
    saved_paths: list[Path] = []
    for offset, edge_size in enumerate(edge_sizes):
        saved_paths.extend(
            generate_and_store_uniform_hypergraph_group(
                edge_size=edge_size,
                count=count,
                seed=seed + offset,
                n_range=n_range,
                weight_range=None,
                average_degree=5.0,
                degree_jitter=1,
                label="small",
            )
        )
    return saved_paths
