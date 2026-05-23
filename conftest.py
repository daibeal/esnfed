"""Make the ``esnfed`` package importable when running pytest from src/python."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Force a headless Matplotlib backend so plotting tests never touch a display.
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass
