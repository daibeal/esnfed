"""Build the GPU-scale FedResPrompt figure + LaTeX table from the collected JSONs.

Reads results/gpu/*.json and writes memoria/figuras/exp12_gpu_scale.pdf and
memoria/tablas/exp12_gpu_scale.tex. FedResPrompt accuracy only (the Federated
LoRA accuracy baseline is left out until a fair, LR-tuned re-run).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
GPU = ROOT / "results" / "gpu"
FIG = ROOT / "memoria" / "figuras"
TAB = ROOT / "memoria" / "tablas"

sweep = json.loads((GPU / "sweep_results.json").read_text())
fair = json.loads((GPU / "results_exp12_fair.json").read_text())  # fair LoRA (lr 1e-4)
lora32 = fair["fedlora"]["acc"]   # the only fairly-tuned LoRA point (32B)

rows = []
for r in sweep:
    if "error" in r:
        continue
    rows.append((r["model"].split("/")[-1], r["params_b"], r["zero_shot"],
                 r["fedresprompt_acc"], r["comm_ratio"]))
# add the 32B run (fair comparison)
rows.append(("Qwen2.5-32B", 32.0, round(fair["fedresprompt"]["zero_shot"], 3),
             round(fair["fedresprompt"]["acc"], 3), round(fair["comm_ratio"], 1)))
rows.sort(key=lambda x: x[1])

# ---- figure: FedResPrompt vs zero-shot accuracy by model -------------------
plt.rcParams.update({"font.size": 10, "figure.dpi": 110})
fig, ax = plt.subplots(figsize=(8.6, 3.8))
labels = [f"{m}\n{p:.1f}B" for m, p, *_ in rows]
x = range(len(rows))
ax.bar(x, [r[3] for r in rows], width=0.6, color="#4C72B0", label="FedResPrompt")
ax.plot(x, [r[2] for r in rows], "o--", color="#C44E52", label="zero-shot")
# the one fairly-tuned Federated LoRA point (32B) -> honest reference
i32 = next(i for i, r in enumerate(rows) if r[1] == 32.0)
ax.plot([i32], [lora32], "D", color="#55A868", markersize=9,
        label="Federated LoRA (tuned, 32B)")
ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("SST-2 test accuracy")
ax.set_title("FedResPrompt steering frozen LLMs (SST-2, non-i.i.d., H200)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "exp12_gpu_scale.pdf", bbox_inches="tight")
print("figure ->", FIG / "exp12_gpu_scale.pdf")

# ---- table -----------------------------------------------------------------
lines = [
    r"\begin{table}[htbp]", r"  \centering",
    r"  \caption{FedResPrompt steering frozen LLMs on SST-2 (non-i.i.d., four "
    r"clients, on an H200 GPU). A fixed reservoir + a tiny trained controller "
    r"reach high accuracy at scale across two model families, communicating "
    r"$8$--$25\times$ fewer floats per round than Federated LoRA while the edge "
    r"never runs the LLM. In the one fairly-tuned head-to-head (32\,B), Federated "
    r"LoRA is \emph{more accurate} ($0.93$ vs.\ $0.83$) but at $25\times$ the "
    r"communication: FedResPrompt trades accuracy for communication/edge "
    r"efficiency, it does not dominate LoRA.}",
    r"  \label{tab:exp12-gpu-scale}",
    r"  \begin{tabular}{lrrrr}", r"    \toprule",
    r"    Model & Params & Zero-shot & FedResPrompt & Comm.\ saving \\",
    r"    \midrule",
]
for m, p, z, a, c in rows:
    lines.append(f"    \\texttt{{{m}}} & {p:.1f}B & {z:.3f} & "
                 f"\\textbf{{{a:.3f}}} & {c:.0f}$\\times$ \\\\")
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
(TAB / "exp12_gpu_scale.tex").write_text("\n".join(lines), encoding="utf-8")
print("table  ->", TAB / "exp12_gpu_scale.tex")
print("\n".join(f"  {m:24s} zero={z:.3f} FRP={a:.3f} comm={c}x" for m, p, z, a, c in rows))
