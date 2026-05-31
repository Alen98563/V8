"""Pytest bootstrap — ensure repo root is importable as the package root.

The V8 dev tree keeps packages at repo root (data/, features/, alpha/, gating/,
orchestrator/, harness/, execution/, monitor/, models/, common/). Adding the repo
root to sys.path lets `from common.engine import ...` resolve both under pytest and
when running `python -m <pkg>.<mod>` from the repo root.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
