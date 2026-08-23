"""Safe loading of training checkpoints (avoids FutureWarning from torch.load default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_training_checkpoint(path: Path | str, map_location: Any = "cpu") -> dict[str, Any]:
    """
    Load a checkpoint saved by ``src.train`` (dict with tensors + metadata).

    Tries ``weights_only=True`` first (PyTorch 2.4+); falls back if the file uses
    structures that require full unpickling.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)
