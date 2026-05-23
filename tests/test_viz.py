"""Tests for the optional visualisation module.

Each backend is skipped if its plotting library is not installed, so the suite
passes whether or not the [viz] extra is present.
"""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, datasets, topologies, viz


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    W = topologies.small_world_reservoir(50, k=6, p=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5, washout=40)
    u, y = datasets.narma10(500, rng=rng)
    esn.fit(u[:350], y[:350])
    pred = esn.predict(u[350:])
    return esn, W, u, y[350:], pred


def _expected_type(backend):
    if backend == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go
        return go.Figure
    pytest.importorskip("matplotlib")
    if backend == "seaborn":
        pytest.importorskip("seaborn")
    from matplotlib.figure import Figure
    return Figure


@pytest.mark.parametrize("backend", ["plotly", "matplotlib", "seaborn"])
def test_plot_reservoir(setup, backend):
    esn, W, *_ = setup
    exp = _expected_type(backend)
    assert isinstance(viz.plot_reservoir(esn, backend=backend), exp)
    # also accepts a raw weight matrix
    assert isinstance(viz.plot_reservoir(W, backend=backend, layout="circular"), exp)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib", "seaborn"])
def test_plot_spectrum(setup, backend):
    esn, W, *_ = setup
    exp = _expected_type(backend)
    assert isinstance(viz.plot_spectrum(W, backend=backend), exp)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib", "seaborn"])
def test_plot_states(setup, backend):
    esn, W, u, *_ = setup
    exp = _expected_type(backend)
    assert isinstance(viz.plot_states(esn, u[:150], n_neurons=5, backend=backend), exp)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib", "seaborn"])
def test_plot_forecast(setup, backend):
    esn, W, u, y_te, pred = setup
    exp = _expected_type(backend)
    assert isinstance(
        viz.plot_forecast(y_te, pred, washout=40, backend=backend), exp
    )


def test_default_backend_is_plotly(setup):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go
    esn, W, *_ = setup
    assert isinstance(viz.plot_spectrum(W), go.Figure)  # no backend kwarg


def test_invalid_backend_raises(setup):
    esn, W, *_ = setup
    with pytest.raises(ValueError):
        viz.plot_spectrum(W, backend="ascii-art")


def test_save_matplotlib_png(setup, tmp_path):
    pytest.importorskip("matplotlib")
    esn, W, *_ = setup
    fig = viz.plot_spectrum(W, backend="matplotlib")
    out = viz.save(fig, str(tmp_path / "spectrum.png"))
    assert (tmp_path / "spectrum.png").exists()


def test_save_plotly_html(setup, tmp_path):
    pytest.importorskip("plotly")
    esn, W, *_ = setup
    fig = viz.plot_reservoir(W, backend="plotly")
    viz.save(fig, str(tmp_path / "reservoir.html"))
    assert (tmp_path / "reservoir.html").exists()
