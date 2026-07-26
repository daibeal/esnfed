"""Package hygiene: exports, import cost, typing and documentation.

These guard the properties a *library* has to keep that no functional test would
notice: that ``import esnfed`` stays cheap and dependency-light, that ``__all__``
tells the truth, and that the declared version is the one that ships.
"""
import importlib
import inspect
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

import esnfed

ROOT = Path(__file__).resolve().parent.parent

# Modules whose only hard dependencies are numpy + networkx.
CORE_MODULES = [
    "esnfed.esn", "esnfed.deep", "esnfed.metrics", "esnfed.topologies",
    "esnfed.datasets", "esnfed.federated", "esnfed.classification",
    "esnfed.privacy", "esnfed.streaming", "esnfed.interop", "esnfed.viz",
    "esnfed.llm_orchestration",
]


class TestExports:

    def test_all_entries_exist(self):
        missing = [name for name in esnfed.__all__ if not hasattr(esnfed, name)]
        assert not missing, f"__all__ names nothing: {missing}"

    def test_every_submodule_is_exported(self):
        submodules = {m.name for m in pkgutil.iter_modules(esnfed.__path__)}
        undeclared = submodules - set(esnfed.__all__)
        assert not undeclared, f"submodules missing from __all__: {undeclared}"

    def test_star_import_matches_all(self):
        ns = {}
        exec("from esnfed import *", ns)
        exported = {k for k in ns if not k.startswith("__")}
        assert exported == set(esnfed.__all__)

    def test_key_entry_points_are_reachable(self):
        for name in ("EchoStateNetwork", "DeepEchoStateNetwork", "nrmse", "rmse",
                     "mse", "mae", "r2_score", "ridge_statistics",
                     "solve_readout"):
            assert callable(getattr(esnfed, name))

    def test_docstring_module_list_matches_the_package(self):
        """The package docstring documents each module; keep it honest."""
        doc = esnfed.__doc__ or ""
        for name in {m.name for m in pkgutil.iter_modules(esnfed.__path__)}:
            assert re.search(rf"^{name}\b", doc, re.MULTILINE), \
                f"module {name!r} is undocumented in the package docstring"


class TestVersion:

    def test_is_valid_semver(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+([-.+].*)?", esnfed.__version__), \
            esnfed.__version__

    def test_hatch_reads_the_version_from_this_attribute(self):
        """pyproject declares a dynamic version sourced from esnfed/__init__.py."""
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'dynamic = ["version"]' in text
        assert 'path = "esnfed/__init__.py"' in text

    def test_version_is_a_single_source_of_truth(self):
        init = (ROOT / "esnfed" / "__init__.py").read_text(encoding="utf-8")
        assert len(re.findall(r"^__version__\s*=", init, re.MULTILINE)) == 1


class TestImportCost:
    """``import esnfed`` must not drag in plotting, SciPy, Numba or pandas.

    The library advertises numpy + networkx as its only runtime requirements, and
    the optional accelerators/backends are imported lazily at the point of use. A
    stray top-level import would break installs that rely on that promise (and
    the browser/Pyodide demo).
    """

    HEAVY = ["matplotlib", "plotly", "seaborn", "scipy", "numba", "pandas",
             "sklearn", "torch", "transformers", "flwr", "reservoirpy"]

    def test_no_heavy_dependency_is_imported(self):
        code = (
            "import sys, json; import esnfed; "
            f"print(json.dumps([m for m in {self.HEAVY!r} if m in sys.modules]))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(ROOT), timeout=180)
        assert out.returncode == 0, out.stderr
        leaked = __import__("json").loads(out.stdout.strip().splitlines()[-1])
        assert leaked == [], f"import esnfed pulled in {leaked}"

    def test_ssl_and_urllib_are_lazy(self):
        """The Pyodide/JupyterLite demo runs without them."""
        code = ("import sys; import esnfed; "
                "print('urllib.request' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(ROOT), timeout=180)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().endswith("False")

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_each_module_imports_standalone(self, module):
        out = subprocess.run([sys.executable, "-c", f"import {module}"],
                             capture_output=True, text=True, cwd=str(ROOT),
                             timeout=180)
        assert out.returncode == 0, out.stderr


class TestTyping:

    def test_py_typed_marker_ships(self):
        assert (ROOT / "esnfed" / "py.typed").exists()

    def test_py_typed_is_included_in_the_wheel_config(self):
        """Hatch includes package data by default; assert the package is listed."""
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'packages = ["esnfed"]' in text

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_modules_use_postponed_annotations(self, module):
        """Required for the ``X | None`` syntax to work on Python 3.9."""
        src = Path(importlib.import_module(module).__file__).read_text(
            encoding="utf-8")
        assert "from __future__ import annotations" in src, module


class TestDocumentation:

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_module_has_a_docstring(self, module):
        assert importlib.import_module(module).__doc__

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_public_callables_are_documented(self, module):
        mod = importlib.import_module(module)
        undocumented = []
        for name, obj in vars(mod).items():
            if name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__module__", None) != module:
                continue        # re-exported from elsewhere
            if not (obj.__doc__ or "").strip():
                undocumented.append(name)
        assert not undocumented, f"{module}: undocumented public API {undocumented}"

    def test_public_classes_document_their_methods(self):
        for cls in (esnfed.EchoStateNetwork, esnfed.DeepEchoStateNetwork):
            for name, member in vars(cls).items():
                if name.startswith("_") or not callable(member):
                    continue
                assert (member.__doc__ or "").strip(), f"{cls.__name__}.{name}"


class TestNoMutableDefaults:
    """A shared mutable default is a classic source of cross-call leakage."""

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_no_list_or_dict_defaults(self, module):
        mod = importlib.import_module(module)
        offenders = []
        for name, obj in vars(mod).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            if getattr(obj, "__module__", None) != module:
                continue
            for param in inspect.signature(obj).parameters.values():
                if isinstance(param.default, (list, dict, set)):
                    offenders.append(f"{name}({param.name}={param.default!r})")
        assert not offenders, offenders
