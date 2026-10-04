"""The benchmark set now lives in the package (fclass.benchdata) so `fclass bench` can use it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fclass.benchdata import CATEGORIES, DATASET  # noqa: E402,F401
