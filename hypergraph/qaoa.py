from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals


def build_hypergraph_hamiltonian(incidence_matrix, lam):
    num_vertices = len(incidence_matrix)
    num_edges = len(incidence_matrix[0]) if num_vertices > 0 else 0
    ham_dict = {}

    #BALANCE TERM
    #Using formula as discussed prev: (n/4)*I + 1/4 * sum_{i != j} Z_i Z_j
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j:
                #Diagonal/identity part: lam * (1/4)
                p_str = "I" * num_vertices
            else:
                #Off-diagonal part Z_i Z_j: lam * (1/4)
                #NOTE: SparsePauliOp uses little-endian
                #thats why we do [num_vertices - 1 - i/j] instead of [i/j]
                p_list = ["I"] * num_vertices
                p_list[num_vertices - 1 - i] = "Z"
                p_list[num_vertices - 1 - j] = "Z"
                p_str = "".join(p_list)

            ham_dict[p_str] = ham_dict.get(p_str, 0.0) + (lam * 0.25)


    #AoN Cut term
    #Small proof to show odd-sized subsets cancel
    #Let k be the number of nodes in edge e.
    # prod(x_i) = prod((I - Z_i)/2) = (1/2^k) * prod(I - Z_i)
    # prod(1 - x_i) = prod((I + Z_i)/2) = (1/2^k) * prod(I + Z_i)
    # The term becomes: 1 - (1/2^k) * [ prod(I - Z_i) + prod(I + Z_i) ]
    # Expansion of prod(I - Z_i): sum of all subsets of Z with sign (-1)^|subset|
    # Expansion of prod(I + Z_i): sum of all subsets of Z with sign (+1)^|subset|
    # Adding them: odd-sized subsets cancel out; even-sized subsets double.
    # Result: 1 - (1/2^k) * [ 2 * sum of even-sized Z-subsets ]
    # Result: 1 - (1/2^(k-1)) * [ sum of even-sized Z-subsets ]

    for e_idx in range(num_edges):

        #putting incidence form into edges
        edge_nodes = []
        for v_idx in range(num_vertices):
            if incidence_matrix[v_idx][e_idx] == 1:
                edge_nodes.append(v_idx)

        k = len(edge_nodes)
        if k == 0: continue

        #even-subset sum coefficient
        coeff = -1.0 / (2**(k - 1))

        #1 constant to identity term
        p_id = "I" * num_vertices
        ham_dict[p_id] = ham_dict.get(p_id, 0.0) + 1.0

        #set of edge_nodes inline to find even-sized subsets
        #(1 << k) represents 2^k
        for i in range(1 << k):
            subset = []
            for j in range(k):
                if (i >> j) & 1:
                    subset.append(edge_nodes[j])

            if len(subset) % 2 == 0:
            #Subset has even cardinality (includes empty set/Identity)
                p_list = ["I"] * num_vertices
                for node_idx in subset:
                    #NOTE: SparsePauliOp uses little-endian
                    #thats why we do [num_vertices - 1 - node_idx] instead of [node_idx]
                    p_list[num_vertices - 1 - node_idx] = "Z"

                p_str = "".join(p_list)
                ham_dict[p_str] = ham_dict.get(p_str, 0.0) + coeff

    return SparsePauliOp.from_list([(k, v) for k, v in ham_dict.items()]).simplify()


def _build_aer_statevector_backend(seed):
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Aer statevector sampling requires qiskit-aer to be installed."
        ) from exc

    return AerSimulator(
        method="statevector",
        max_parallel_threads=0,
        seed_simulator=seed,
    )


def _build_aer_statevector_sampler(seed, backend=None):
    from qiskit.primitives import BackendSamplerV2

    if backend is None:
        backend = _build_aer_statevector_backend(seed)

    try:
        return BackendSamplerV2(backend=backend)
    except TypeError:
        return BackendSamplerV2(backend)


def _build_aer_qaoa_runtime(seed):
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError as exc:
        raise ImportError(
            "Aer-backed QAOA transpilation requires Qiskit's preset pass managers."
        ) from exc

    backend = _build_aer_statevector_backend(seed)
    sampler = _build_aer_statevector_sampler(seed, backend=backend)
    transpiler = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
    )
    return sampler, transpiler


def run_hypergraph_qaoa(
    incidence_matrix,
    lam=1.0,
    reps=1,
    num_partitions=10,
    optimizer_options=None,
    seed=42,
):
    result = run_hypergraph_qaoa_result(
        incidence_matrix,
        lam=lam,
        reps=reps,
        optimizer_options=optimizer_options,
        seed=seed,
    )
    return _sample_top_k_partitions(result.eigenstate, num_partitions)


def run_hypergraph_qaoa_result(
    incidence_matrix,
    lam=1.0,
    reps=1,
    optimizer_options=None,
    seed=42,
):
    hamiltonian = build_hypergraph_hamiltonian(incidence_matrix, lam)

    algorithm_globals.random_seed = seed
    sampler, transpiler = _build_aer_qaoa_runtime(seed)

    optimizer = COBYLA(**(optimizer_options or {}))
    qaoa = QAOA(sampler, optimizer, reps=reps, transpiler=transpiler)
    return qaoa.compute_minimum_eigenvalue(hamiltonian)


def _sample_most_likely(quasi_distribution):
    bitstring = max(quasi_distribution.items(), key=lambda item: item[1])[0]
    return [int(bit) for bit in bitstring[::-1]]


def _sample_top_k_partitions(quasi_distribution, num_partitions):
    top_bitstrings = sorted(
        quasi_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:num_partitions]
    return [[int(bit) for bit in bitstring[::-1]] for bitstring, _ in top_bitstrings]


def sample_top_k_states(quasi_distribution, num_partitions):
    top_bitstrings = sorted(
        quasi_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:num_partitions]
    return [
        {
            "rank": rank,
            "bitstring": bitstring,
            "probability": float(probability),
            "partition": [int(bit) for bit in bitstring[::-1]],
        }
        for rank, (bitstring, probability) in enumerate(top_bitstrings, start=1)
    ]
