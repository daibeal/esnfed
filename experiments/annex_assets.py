"""Generate annex-only assets from already-saved results.

Reads ``results/gpu/sweep_results.json`` (the multi-model GPU sweep) and emits:

* ``memoria/figuras/gpu_sweep.pdf`` --- per-model zero-shot vs FedResPrompt
  accuracy on SST-2, showing the lift a fixed reservoir + tiny controller adds to
  each frozen backbone; and
* ``memoria/tablas/gpu_sweep.tex`` --- the same sweep as a table, with the model
  sizes, VRAM and load times measured on the rented H200.

Nothing here is recomputed: it only re-renders data already committed to the
repository, so it is safe to run on a CPU with no GPU.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from common import FIGURES, REPO_ROOT, TABLES, save_figure, set_style

SWEEP = REPO_ROOT / "results" / "gpu" / "sweep_results.json"


def _short(model: str) -> str:
    """A compact display name for a HuggingFace model id."""
    name = model.split("/")[-1]
    return name.replace("-instruct", "").replace("Qwen2.5", "Qwen2.5")


def load_rows() -> list[dict]:
    data = json.loads(SWEEP.read_text(encoding="utf-8"))
    rows = [d for d in data if "error" not in d]
    rows.sort(key=lambda d: d["params_b"])
    return rows


def make_figure(rows: list[dict]) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = [d["params_b"] for d in rows]
    zs = [d["zero_shot"] for d in rows]
    fp = [d["fedresprompt_acc"] for d in rows]

    # vertical connector showing the lift per model
    for x, z, f in zip(xs, zs, fp):
        ax.plot([x, x], [z, f], color="0.6", lw=1.0, zorder=1)
    ax.scatter(xs, zs, s=55, color="#999999", label="zero-shot (frozen)", zorder=2)
    ax.scatter(xs, fp, s=55, color="#4C72B0", label="FedResPrompt", zorder=3)

    for d in rows:
        ax.annotate(_short(d["model"]), (d["params_b"], max(d["zero_shot"], d["fedresprompt_acc"])),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5,
                    rotation=0)

    ax.axhline(0.5, color="0.5", ls=":", lw=1.0)
    ax.text(rows[0]["params_b"], 0.505, "chance", fontsize=8, color="0.4", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("model size (billion parameters, log scale)")
    ax.set_ylabel("SST-2 accuracy (non-i.i.d., 4 clients)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right")
    save_figure(fig, "gpu_sweep")


def make_table(rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{GPU model-zoo sweep on SST-2 (non-i.i.d., four clients, one"
        r" H200). Per model: size, peak VRAM and load time measured on the rented"
        r" instance, zero-shot accuracy of the frozen backbone, FedResPrompt"
        r" accuracy, and the client--server communication saving over Federated"
        r" LoRA. The per-model LoRA accuracies (taken at the controller's learning"
        r" rate, not separately tuned) are in \texttt{results/gpu/}; the fair,"
        r" LR-tuned head-to-head is the 32\,B comparison in \cref{sec:fedresprompt}."
        r" Phi-3.5-mini failed to load under the pinned stack.}",
        r"  \label{tab:gpu-sweep}",
        r"  \fitwidth{\begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Model & Params (B) & VRAM (GB) & Load (s) & Zero-shot & FedResPrompt"
        r" & Comm.\ saving \\",
        r"    \midrule",
    ]
    for d in rows:
        lines.append(
            f"    \\texttt{{{_short(d['model'])}}} & {d['params_b']:.2f} & "
            f"{d['vram_gb']:.1f} & {d['load_s']:.1f} & {d['zero_shot']:.3f} & "
            f"\\textbf{{{d['fedresprompt_acc']:.3f}}} & {round(d['comm_ratio'])}$\\times$ \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]
    out = TABLES / "gpu_sweep.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  latex  -> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    rows = load_rows()
    make_figure(rows)
    make_table(rows)
    print(f"  ({len(rows)} models; figures -> {FIGURES.name}, tables -> {TABLES.name})")


if __name__ == "__main__":
    main()
