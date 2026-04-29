from __future__ import annotations

import csv
import math
import sys
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypergraph import annealer, evaluation, loader, qaoa  # noqa: E402

STEM = "qaoa_sa_comparison"
N_VALUES = range(8, 16)
LAMBDAS = (0.3, 1.0, 3.0)
RUN_SEEDS = (42, 43, 44, 45, 46)
INSTANCES_PER_N = 100
NUM_QAOA_PARTITIONS = 10
ANNEALER_NUM_READS = 100
LOG_DIR = ROOT / "logs"


def lambda_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def partition_bitstring(partition: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in np.asarray(partition, dtype=int)[::-1])


def enumerate_balanced_optima(H: np.ndarray) -> tuple[float, set[str]]:
    num_vertices = H.shape[1]
    best_score = float("inf")
    optimal_bitstrings: set[str] = set()
    for rest in product((0, 1), repeat=num_vertices - 1):
        partition = np.array((0, *rest), dtype=int)
        if not evaluation.is_balanced(partition, 2):
            continue
        score = evaluation.hypergraph_cut(partition, H, objective=evaluation.AON)
        if score < best_score:
            best_score = float(score)
            optimal_bitstrings = {partition_bitstring(partition), partition_bitstring(1 - partition)}
        elif np.isclose(score, best_score):
            optimal_bitstrings.add(partition_bitstring(partition))
            optimal_bitstrings.add(partition_bitstring(1 - partition))
    return best_score, optimal_bitstrings


def ratio_or_blank(value: float | None, optimal_value: float) -> str:
    if value is None:
        return ""
    if np.isclose(optimal_value, 0.0):
        return "1.000000" if np.isclose(value, optimal_value) else ""
    return f"{value / optimal_value:.6f}"


def summarize_run_rates(run_counts: list[int], denominators: list[int]) -> tuple[str, str]:
    rates = [count / denom for count, denom in zip(run_counts, denominators, strict=True) if denom > 0]
    if not rates:
        return "", ""
    return f"{mean(rates):.6f}", f"{float(np.std(rates)):.6f}"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for n in N_VALUES:
        instance_names = [f"hypergraph_r3_n{n}_d5_w1_{i}" for i in range(INSTANCES_PER_N)]
        print(f"n={n}: {len(instance_names)} instances")
        qaoa_stats = {lam: {"attempts": [0] * 5, "feasible": [0] * 5, "optimal": [0] * 5, "ratios": [[] for _ in RUN_SEEDS], "times": []} for lam in LAMBDAS}
        sa_stats = {lam: {"attempts": [0] * 5, "feasible": [0] * 5, "optimal": [0] * 5, "ratios": [[] for _ in RUN_SEEDS], "times": []} for lam in LAMBDAS}

        for instance_name in instance_names:
            H = loader.load_hypergraph(instance_name)
            optimal_value, optimal_bitstrings = enumerate_balanced_optima(H)
            for lam in LAMBDAS:
                for run_index, seed in enumerate(RUN_SEEDS):
                    start = perf_counter()
                    result = qaoa.run_hypergraph_qaoa_result(H.T, lam=lam, seed=seed)
                    elapsed = perf_counter() - start
                    best_value = None
                    best_partition = None
                    for state in qaoa.sample_top_k_states(result.eigenstate, NUM_QAOA_PARTITIONS):
                        partition = np.asarray(state["partition"], dtype=int)
                        metrics = evaluation.evaluate_partition(partition, H, k=2, verbose=False)
                        if metrics["balanced"] and (best_value is None or float(metrics["aon"]) < best_value):
                            best_value = float(metrics["aon"])
                            best_partition = partition
                    feasible = best_value is not None
                    optimal = bool(feasible and np.isclose(best_value, optimal_value))
                    stats = qaoa_stats[lam]
                    stats["attempts"][run_index] += 1
                    stats["feasible"][run_index] += int(feasible)
                    stats["optimal"][run_index] += int(optimal)
                    stats["times"].append(elapsed)
                    approx = ratio_or_blank(best_value, optimal_value)
                    if approx:
                        stats["ratios"][run_index].append(float(approx))
                    raw_rows.append({"n": n, "instance": instance_name, "method": "qaoa", "lambda": lambda_label(lam), "run_index": run_index, "seed": seed, "feasible": feasible, "optimal": optimal, "objective_value": "" if best_value is None else f"{best_value:.6f}", "optimal_value": f"{optimal_value:.6f}", "approximation_ratio": approx, "runtime_s": f"{elapsed:.6f}"})

                    start = perf_counter()
                    partition = annealer.run_hypergraph_simulated_annealing(H, k=2, lam=lam, num_reads=ANNEALER_NUM_READS, seed=seed)
                    elapsed = perf_counter() - start
                    metrics = evaluation.evaluate_partition(partition, H, k=2, verbose=False)
                    feasible = bool(metrics["balanced"])
                    value = float(metrics["aon"]) if feasible else None
                    optimal = bool(feasible and np.isclose(value, optimal_value))
                    stats = sa_stats[lam]
                    stats["attempts"][run_index] += 1
                    stats["feasible"][run_index] += int(feasible)
                    stats["optimal"][run_index] += int(optimal)
                    stats["times"].append(elapsed)
                    approx = ratio_or_blank(value, optimal_value)
                    if approx:
                        stats["ratios"][run_index].append(float(approx))
                    raw_rows.append({"n": n, "instance": instance_name, "method": "annealer", "lambda": lambda_label(lam), "run_index": run_index, "seed": seed, "feasible": feasible, "optimal": optimal, "objective_value": "" if value is None else f"{value:.6f}", "optimal_value": f"{optimal_value:.6f}", "approximation_ratio": approx, "runtime_s": f"{elapsed:.6f}"})

        for method, all_stats in (("qaoa", qaoa_stats), ("annealer", sa_stats)):
            for lam in LAMBDAS:
                stats = all_stats[lam]
                feasibility_rate, feasibility_std = summarize_run_rates(stats["feasible"], stats["attempts"])
                optimality_rate, optimality_std = summarize_run_rates(stats["optimal"], stats["feasible"])
                ratio_means = [mean(ratios) for ratios in stats["ratios"] if ratios]
                ratio, ratio_std = ("", "") if not ratio_means else (f"{mean(ratio_means):.6f}", f"{float(np.std(ratio_means)):.6f}")
                summary_rows.append({"n": n, "edge_size": 3, "average_degree": 5, "method": method, "lambda": lambda_label(lam), "instances": INSTANCES_PER_N, "runs_per_instance": len(RUN_SEEDS), "optimality_rate": optimality_rate, "optimality_rate_std": optimality_std, "feasibility_rate": feasibility_rate, "feasibility_rate_std": feasibility_std, "approximation_ratio": ratio, "approximation_ratio_std": ratio_std, "avg_runtime_s": f"{mean(stats['times']):.6f}", "notes": "Regenerated from committed instances"})

    raw_path = LOG_DIR / f"{STEM}_raw.csv"
    summary_path = LOG_DIR / f"{STEM}_summary.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {raw_path}")


if __name__ == "__main__":
    main()
