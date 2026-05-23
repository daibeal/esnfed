"""Shared utilities for the experiment scripts: paths, plotting style, I/O.

Figures are written to ``memoria/figuras`` (tracked, embedded in the thesis),
generated LaTeX tables to ``memoria/tablas`` (tracked), and raw CSV results to
``salidas`` (gitignored, regenerable).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import pandas as pd

# Make the esnfed package importable regardless of the working directory.
PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

def _find_root(start: Path):
    """Locate the output root, working in both layouts:

    * the private thesis monorepo (a parent contains ``memoria/``), where outputs
      go into the thesis tree; and
    * the standalone public ``esnfed`` repo (nearest ``pyproject.toml``), where
      outputs go into local ``figures/``, ``tables/`` and ``outputs/`` folders.
    """
    for p in [start, *start.parents]:
        if (p / "memoria").is_dir():
            return p, True
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p, False
    return start.parents[2], False


REPO_ROOT, _IS_THESIS = _find_root(Path(__file__).resolve())
if _IS_THESIS:
    FIGURES = REPO_ROOT / "memoria" / "figuras"
    TABLES = REPO_ROOT / "memoria" / "tablas"
    OUTPUTS = REPO_ROOT / "salidas"
else:
    FIGURES = REPO_ROOT / "figures"
    TABLES = REPO_ROOT / "tables"
    OUTPUTS = REPO_ROOT / "outputs"
for _d in (FIGURES, TABLES, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibility: a single master seed reused across experiments.
MASTER_SEED = 20260523

# Consistent colours for topologies / strategies across all figures.
PALETTE = {
    "random": "#4C72B0",
    "small_world": "#DD8452",
    "scale_free": "#55A868",
    "ring": "#C44E52",
    "centralized": "#000000",
    "federated_ridge": "#4C72B0",
    "fedavg": "#DD8452",
    "ensemble": "#55A868",
    "local": "#C44E52",
    "alignment": "#8172B3",
}

LABELS = {
    "random": "Erdos-Renyi",
    "small_world": "Small-world",
    "scale_free": "Scale-free",
    "ring": "Ring",
    "centralized": "Centralized (pooled)",
    "federated_ridge": "Federated ridge (exact)",
    "fedavg": "FedAvg (iterative)",
    "ensemble": "Ensemble (heterogeneous)",
    "local": "Local-only (mean)",
}


def set_style() -> None:
    """Apply a clean, publication-quality matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "figure.autolayout": True,
        }
    )


def save_figure(fig, name: str) -> Path:
    """Save a figure as PDF (for LaTeX) into memoria/figuras."""
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path.relative_to(REPO_ROOT)}")
    return path


def save_table(df: pd.DataFrame, name: str, **to_csv_kwargs) -> Path:
    """Save a results table as CSV into salidas."""
    path = OUTPUTS / f"{name}.csv"
    df.to_csv(path, index=False, **to_csv_kwargs)
    print(f"  table  -> {path.relative_to(REPO_ROOT)}")
    return path


def _fmt_cell(x, float_format: str) -> str:
    """Format one cell; numbers via float_format, everything else as text."""
    if isinstance(x, float):
        if x != x:  # NaN
            return "--"
        return float_format % x
    if hasattr(x, "item"):  # numpy scalar
        try:
            return float_format % float(x)
        except (TypeError, ValueError):
            return str(x)
    return str(x)


def _is_integer_column(series: pd.Series) -> bool:
    """True if every value is a whole number (so it should print without decimals)."""
    try:
        vals = pd.to_numeric(series)
    except (ValueError, TypeError):
        return False
    return bool((vals.dropna() % 1 == 0).all()) and len(vals.dropna()) > 0


def save_latex_table(df: pd.DataFrame, name: str, caption: str, label: str,
                     float_format: str = "%.4f", column_format: str | None = None) -> Path:
    """Write a booktabs LaTeX table into memoria/tablas for \\input{} (no deps).

    Column headers are emitted verbatim, so math/LaTeX in column names is kept.
    Columns whose values are all whole numbers are printed as integers.
    """
    path = TABLES / f"{name}.tex"
    ncol = df.shape[1]
    colspec = column_format or ("l" + "r" * (ncol - 1))
    int_cols = {c for c in df.columns if _is_integer_column(df[c])}
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for col, v in row.items():
            if col in int_cols and isinstance(v, (int, float)) and v == v:
                cells.append(str(int(round(v))))
            else:
                cells.append(_fmt_cell(v, float_format))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  latex  -> {path.relative_to(REPO_ROOT)}")
    return path
