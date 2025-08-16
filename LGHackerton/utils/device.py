"""Utilities for selecting the computation device.

This module tries to import :mod:`torch` but falls back to ``None`` if the
library is not available.  Downstream code can therefore still run on
environments where PyTorch is not installed (e.g. when only using the LightGBM
model).
"""

from __future__ import annotations

import os

try:  # pragma: no cover - best effort in absence of torch
    import torch
except Exception:  # torch is optional, treat as unavailable if import fails
    torch = None


def select_device(default: str | None = None) -> str:
    """Select a computation device.

    The function interacts with the user to select ``'cpu'``, ``'cuda'`` or
    ``'mps'`` (Apple Metal) when no default is provided.  If a default device is
    supplied via the ``default`` argument or the ``DEVICE`` environment
    variable, that value is returned immediately without prompting.  When
    running in a non-interactive environment where ``input`` raises
    :class:`EOFError`, the ``default``/environment variable value is returned or
    ``'cpu'`` if none was given.
    """

    default_device = default or os.environ.get("DEVICE")
    if default_device:
        return default_device

    try:
        while True:
            choice = input("Select compute environment (macOS/gpu/cpu): ").strip().lower()
            if choice == "macos":
                if torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    return "mps"
                print("MPS not available. Please choose another option.")
            elif choice in ("gpu", "cuda"):
                if torch and torch.cuda.is_available():
                    return "cuda"
                print("CUDA GPU not available. Please choose another option.")
            elif choice == "cpu":
                return "cpu"
            else:
                print("Invalid option. Choose from macOS/gpu/cpu.")
    except EOFError:
        # When running without a TTY, return the safest option.
        return "cpu"
