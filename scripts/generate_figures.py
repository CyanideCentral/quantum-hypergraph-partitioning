from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "qaoa_sa_comparison"
SUMMARY_CSV = ROOT / "logs" / f"{STEM}_summary.csv"
FIGURE_DIR = ROOT / "figures" / STEM
FIGURE_TEX = FIGURE_DIR / "figures.tex"
N_VALUES = range(8, 16)
LAMBDAS = (0.3, 1.0, 3.0)
RUNS_PER_INSTANCE = 5


def lambda_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def metric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def series_styles(method: str) -> dict[float, str]:
    markers = ("o", "square*", "triangle*")
    if method == "qaoa":
        colors = ("blue", "blue!80!black", "blue!65!black")
        prefix = ""
    else:
        colors = ("red", "red!80!black", "red!65!black")
        prefix = "dashed, "
    return {
        lam: f"{colors[i]}, {prefix}mark={markers[i]}, mark options={{solid, fill=white}}"
        for i, lam in enumerate(LAMBDAS)
    }


def coordinates(rows: dict[tuple[int, str, str], dict[str, str]], method: str, lam: float, key: str, std_key: str) -> str:
    values = []
    label = lambda_label(lam)
    for n in N_VALUES:
        row = rows[(n, method, label)]
        y = metric(row, key)
        se = metric(row, std_key) / math.sqrt(RUNS_PER_INSTANCE)
        values.append(f"    ({n},{y:.6f}) +- (0,{se:.6f})")
    return "\n".join(values)


def axis(rows: dict[tuple[int, str, str], dict[str, str]], key: str, std_key: str, ylabel: str, caption: str, label: str) -> list[str]:
    qaoa_styles = series_styles("qaoa")
    sa_styles = series_styles("sa")
    plots: list[str] = []
    for lam in LAMBDAS:
        plots.extend([
            rf"\addplot+[{qaoa_styles[lam]}, error bars/.cd, y dir=both, y explicit] coordinates {{",
            coordinates(rows, "qaoa", lam, key, std_key),
            "};",
        ])
    for lam in LAMBDAS:
        plots.extend([
            rf"\addplot+[{sa_styles[lam]}, error bars/.cd, y dir=both, y explicit] coordinates {{",
            coordinates(rows, "annealer", lam, key, std_key),
            "};",
        ])

    return [
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=\linewidth,",
        r"    height=0.52\linewidth,",
        r"    xlabel={Number of nodes $n$},",
        rf"    ylabel={{{ylabel}}},",
        r"    xmin=8, xmax=15,",
        r"    ymin=0, ymax=1.05,",
        r"    xtick={8,9,10,11,12,13,14,15},",
        r"    grid=both,",
        r"    grid style={gray!20},",
        r"    major grid style={gray!35},",
        r"    line width=1pt,",
        r"    mark size=2.2pt,",
        r"]",
        "",
        *plots,
        "",
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ]


def latex_source(summary_rows: list[dict[str, str]]) -> str:
    rows = {(int(r["n"]), r["method"], r["lambda"]): r for r in summary_rows}
    qaoa_styles = series_styles("qaoa")
    sa_styles = series_styles("sa")
    legend: list[str] = []
    for lam in LAMBDAS:
        legend.extend([
            rf"\addlegendimage{{{qaoa_styles[lam]}, line width=1pt}}",
            rf"\addlegendentry{{\textsc{{sQAOA}}, $\lambda={lambda_label(lam)}$}}",
        ])
    for lam in LAMBDAS:
        legend.extend([
            rf"\addlegendimage{{{sa_styles[lam]}, line width=1pt}}",
            rf"\addlegendentry{{\textsc{{SA}}, $\lambda={lambda_label(lam)}$}}",
        ])

    return "\n".join([
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{tikz}",
        r"\usepackage{pgfplots}",
        "",
        r"\pgfplotsset{compat=1.18}",
        "",
        r"\begin{document}",
        "",
        r"% ---------- Shared legend ----------",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    hide axis,",
        r"    xmin=0, xmax=1,",
        r"    ymin=0, ymax=1,",
        r"    legend columns=3,",
        r"    legend style={draw=none, fill=none, font=\small, column sep=0.35em}",
        r"]",
        *legend,
        r"\end{axis}",
        r"\end{tikzpicture}",
        "",
        *axis(rows, "optimality_rate", "optimality_rate_std", "Optimality rate", "Optimality rate with error bars showing standard error across five runs.", "fig:r3-d5-100-optimality"),
        "",
        *axis(rows, "feasibility_rate", "feasibility_rate_std", "Feasibility rate", "Feasibility rate with error bars showing standard error across five runs.", "fig:r3-d5-100-feasibility"),
        "",
        r"\end{document}",
        "",
    ])


def main() -> None:
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_TEX.write_text(latex_source(summary_rows), encoding="utf-8")
    print(f"Wrote {FIGURE_TEX}")


if __name__ == "__main__":
    main()
