# Installation

`esnfed` runs on **Python 3.9+**. The core has only two dependencies (NumPy and
NetworkX); everything heavier is an optional *extra*.

```bash
pip install esnfed                  # core (numpy + networkx)
```

## Optional extras

| Extra | Adds | For |
|-------|------|-----|
| `fast` | numba, scipy | optional [acceleration](performance.md) (Numba JIT, sparse reservoirs) |
| `viz` | plotly, matplotlib, seaborn, kaleido | the `esnfed.viz` plots |
| `experiments` | matplotlib, pandas, scipy, scikit-learn | reproducing the experiments |
| `reservoirpy` | reservoirpy | designing reservoirs in ReservoirPy |
| `flower` | flwr | the Flower integration example |
| `llm` | transformers | the experimental FedResPrompt LLM backend |
| `dev` | pytest, build, twine | development |

```bash
pip install "esnfed[fast]"                   # Numba + sparse acceleration
pip install "esnfed[viz]"                    # plotting
pip install "esnfed[viz,experiments]"        # plotting + data-science stack
pip install "esnfed[reservoirpy,flower,llm]" # all integrations
```

!!! note "The `llm` extra also needs a torch build"
    The FedResPrompt real-LLM backend (`TransformersLM`) needs `torch` in
    addition to `transformers`. The default surrogate language model has no such
    requirement.

## From source

```bash
git clone https://github.com/daibeal/esnfed.git
cd esnfed
pip install -e ".[viz,experiments,dev]"
python -m pytest        # 59 tests
```

## Verify

```python
import esnfed
print(esnfed.__version__)
```
