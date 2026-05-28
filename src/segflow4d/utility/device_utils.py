"""
Device abstraction for cross-backend torch operations.

Centralises device detection (cuda / mps / cpu), device-string formatting,
synchronisation, cache-clearing, and the context-manager idiom so the
rest of the codebase can stop hard-coding ``cuda``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceKind = Literal["cuda", "mps", "cpu"]


def detect_device_kind() -> DeviceKind:
    """Return the preferred accelerator kind, falling back to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def device_count(kind: DeviceKind | None = None) -> int:
    kind = kind or detect_device_kind()
    if kind == "cuda":
        return torch.cuda.device_count()
    if kind == "mps":
        return 1
    return 0


def device_str(device_id: int = 0, kind: DeviceKind | None = None) -> str:
    """Return the torch device string (e.g. ``cuda:0``, ``mps``, ``cpu``)."""
    kind = kind or detect_device_kind()
    if kind == "cuda":
        return f"cuda:{device_id}"
    if kind == "mps":
        return "mps"
    return "cpu"


def is_accelerator_available() -> bool:
    return detect_device_kind() in ("cuda", "mps")


def set_device(device_id: int = 0, kind: DeviceKind | None = None) -> None:
    """Set the current device. No-op outside CUDA (MPS has a single device)."""
    kind = kind or detect_device_kind()
    if kind == "cuda":
        torch.cuda.set_device(device_id)


def current_device_id(kind: DeviceKind | None = None) -> int:
    kind = kind or detect_device_kind()
    if kind == "cuda":
        return torch.cuda.current_device()
    return 0


def device_context(device_id: int = 0, kind: DeviceKind | None = None):
    """Context-manager analogue of ``torch.cuda.device(id)``; no-op on MPS/CPU."""
    kind = kind or detect_device_kind()
    if kind == "cuda":
        return torch.cuda.device(device_id)
    return contextlib.nullcontext()


def synchronize(device_id: int = 0, kind: DeviceKind | None = None) -> None:
    kind = kind or detect_device_kind()
    if kind == "cuda":
        torch.cuda.synchronize(device_id)
    elif kind == "mps":
        torch.mps.synchronize()


def empty_cache(kind: DeviceKind | None = None) -> None:
    kind = kind or detect_device_kind()
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            # Older torch versions
            pass
