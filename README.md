# Quantum Hypergraph Partitioning

This repository contains the code and data needed to reproduce the hypergraph partitioning experiments used for the plots in our paper, *Quantum Hypergraph Partitioning* (Q-Data '26).

## Contents

- `data/`: randomly generated hypergraph instances, 100 each for `n=8,...,15`.
- `logs/*_summary.csv`: summarized QAOA/SA results used by the plots.
- `logs/*_raw.csv`: per-instance, per-run result records.
- `scripts/generate_figures.py`: regenerates the LaTeX source for the optimality and feasibility plots from the summary CSV.
- `scripts/run_experiment.py`: reruns QAOA and simulated annealing on all instances from the dataset.

## Reproducing the results

```bash
uv sync
uv run python scripts/run_experiment.py
uv run python scripts/generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory figures/qaoa_sa_comparison figures/qaoa_sa_comparison/figures.tex
```

This reruns the experiments and writes `figures/qaoa_sa_comparison/figures.tex` containing the optimality-rate and feasibility-rate figures.

